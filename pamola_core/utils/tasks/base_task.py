"""
Base Task Class for HHR project.

This module provides the foundation for all task implementations in the project,
defining the standard lifecycle, configuration handling, and operation management.
"""

import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from pamola_core.utils.io import ensure_directory
from pamola_core.utils.logging import configure_task_logging
from pamola_core.utils.ops.op_base import OperationScope
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import create_operation_instance
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.tasks.task_config import load_task_config
from pamola_core.utils.tasks.task_registry import register_task_execution, check_task_dependencies
from pamola_core.utils.tasks.task_reporting import TaskReporter


class BaseTask:
    """
    Base class for all tasks.

    Defines the interface and common functionality for all tasks,
    including lifecycle, configuration handling, and operation interaction.
    """

    def __init__(self,
                 task_id: str,
                 task_type: str,
                 description: str,
                 input_datasets: Dict[str, str] = None,
                 auxiliary_datasets: Dict[str, str] = None):
        """
        Initialize the task.

        Parameters:
        -----------
        task_id : str
            Unique identifier for the task
        task_type : str
            Type of task (profiling, anonymization, etc.)
        description : str
            Description of the task's purpose
        input_datasets : Dict[str, str], optional
            Dictionary mapping dataset names to file paths
        auxiliary_datasets : Dict[str, str], optional
            Dictionary mapping auxiliary dataset names to file paths
        """
        self.task_id = task_id
        self.task_type = task_type
        self.description = description
        self.input_datasets = input_datasets or {}
        self.auxiliary_datasets = auxiliary_datasets or {}

        # These will be initialized later
        self.config = None
        self.logger = None
        self.reporter = None
        self.task_dir = None
        self.data_source = None
        self.operations = []
        self.start_time = None

        # New fields for enhanced functionality
        self.encryption_key = None
        self.use_encryption = False
        self.use_vectorization = False
        self.parallel_processes = 1
        self.global_scope = None

    def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the task: load configuration, create directories,
        set up logging, check dependencies.

        Parameters:
        -----------
        args : Dict[str, Any], optional
            Command line arguments to override configuration

        Returns:
        --------
        bool
            True if initialization is successful, False otherwise
        """
        try:
            # Record start time
            self.start_time = time.time()

            # Load configuration
            self.config = load_task_config(self.task_id, self.task_type, args)

            # Create task directory
            self.task_dir = Path(self.config.output_directory)
            ensure_directory(self.task_dir)

            # Set up logging
            self.logger = configure_task_logging(self.task_dir, self.config.log_level)
            self.logger.info(f"Initializing task: {self.task_id} ({self.description})")

            # Create reporter
            self.reporter = TaskReporter(
                self.task_id,
                self.task_type,
                self.description,
                self.config.report_path
            )

            # Check dependencies
            if not check_task_dependencies(self.task_id, self.task_type, self.config.dependencies):
                self.logger.error(f"Task dependencies not satisfied for {self.task_id}")
                return False

            # Initialize encryption settings
            self.use_encryption = getattr(self.config, "use_encryption", False)
            if self.use_encryption:
                self._initialize_encryption()

            # Initialize vectorization settings
            self.use_vectorization = getattr(self.config, "use_vectorization", False)
            self.parallel_processes = getattr(self.config, "parallel_processes", 1)

            # Initialize global scope if provided in configuration
            self._initialize_global_scope()

            # Create data source
            self._initialize_data_source()

            return True
        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error initializing task {self.task_id}: {str(e)}")
            else:
                logging.exception(f"Error initializing task {self.task_id}: {str(e)}")
            return False

    def _initialize_encryption(self):
        """
        Initialize encryption settings by loading encryption key.
        """
        encryption_key_path = getattr(self.config, "encryption_key_path", None)
        if encryption_key_path:
            try:
                key_path = Path(encryption_key_path)
                if key_path.exists():
                    with open(key_path, 'rb') as f:
                        self.encryption_key = f.read()
                    self.logger.info(f"Loaded encryption key from {encryption_key_path}")
                else:
                    self.logger.warning(f"Encryption key file {encryption_key_path} not found, disabling encryption")
                    self.use_encryption = False
            except Exception as e:
                self.logger.error(f"Error loading encryption key: {str(e)}, disabling encryption")
                self.use_encryption = False
        else:
            self.logger.warning("Encryption enabled but no key path provided, disabling encryption")
            self.use_encryption = False

    def _initialize_global_scope(self):
        """
        Initialize global operation scope from configuration.
        """
        scope_config = getattr(self.config, "scope", {})
        if scope_config:
            fields = scope_config.get("fields", [])
            datasets = scope_config.get("datasets", [])
            field_groups = scope_config.get("field_groups", {})

            self.global_scope = OperationScope(
                fields=fields,
                datasets=datasets,
                field_groups=field_groups
            )
            self.logger.info(f"Initialized global scope with {len(fields)} fields, "
                             f"{len(datasets)} datasets, and {len(field_groups)} field groups")

    def _initialize_data_source(self):
        """
        Initialize the data source with input and auxiliary datasets.
        """
        self.data_source = DataSource()

        # Process input datasets
        for name, path in self.input_datasets.items():
            self.logger.info(f"Adding input dataset: {name} from {path}")
            self.data_source.add_file_path(name, path)

        # Process auxiliary datasets
        for name, path in self.auxiliary_datasets.items():
            self.logger.info(f"Adding auxiliary dataset: {name} from {path}")
            self.data_source.add_file_path(name, path)

    def configure_operations(self):
        """
        Configure operations to be executed by the task.
        Should be overridden in subclasses.
        """
        raise NotImplementedError("Subclasses must implement configure_operations()")

    def execute(self) -> bool:
        """
        Execute the task: run operations sequentially,
        collect results, generate report.

        Returns:
        --------
        bool
            True if execution is successful, False otherwise
        """
        self.logger.info(f"Executing task: {self.task_id}")

        # Check if we have operations configured
        if not self.operations:
            self.logger.error("No operations configured for this task")
            return False

        # Create progress tracker for the task
        with ProgressTracker(
                total=len(self.operations),
                description=f"Task: {self.task_id}",
                unit="operations"
        ) as task_progress:

            # Execute operations sequentially
            for i, operation in enumerate(self.operations):
                operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"
                self.logger.info(f"Executing operation: {operation_name}")

                task_progress.update(0, {"operation": operation_name, "status": "running"})

                try:
                    # Execute the operation
                    result = operation.run(
                        data_source=self.data_source,
                        task_dir=self.task_dir,
                        reporter=self.reporter,
                        track_progress=True,
                        parallel_processes=self.parallel_processes if self.use_vectorization else 1,
                        use_encryption=self.use_encryption,
                        encryption_key=self.encryption_key
                    )

                    # Check result status
                    if result.status.name == "ERROR":
                        self.logger.error(f"Operation {operation_name} failed: {result.error_message}")

                        # Check if we should continue on error
                        if not self.config.continue_on_error:
                            self.logger.error("Aborting task due to operation failure")
                            return False

                    task_progress.update(1, {
                        "operation": operation_name,
                        "status": result.status.name
                    })

                except Exception as e:
                    self.logger.exception(f"Error executing operation {operation_name}: {str(e)}")

                    # Check if we should continue on error
                    if not self.config.continue_on_error:
                        self.logger.error("Aborting task due to operation failure")
                        return False

                    task_progress.update(1, {
                        "operation": operation_name,
                        "status": "ERROR"
                    })

        return True

    def finalize(self, success: bool) -> bool:
        """
        Finalize the task: release resources, close files,
        register execution result.

        Parameters:
        -----------
        success : bool
            Whether the task executed successfully

        Returns:
        --------
        bool
            True if finalization is successful, False otherwise
        """
        try:
            # Calculate execution time
            execution_time = time.time() - self.start_time

            self.logger.info(f"Finalizing task: {self.task_id} (success: {success})")
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")

            # Add final status to reporter
            self.reporter.add_task_summary(
                success=success,
                execution_time=execution_time
            )

            # Generate and save report
            report_path = self.reporter.save_report()
            self.logger.info(f"Task report saved to: {report_path}")

            # Register task execution
            register_task_execution(
                self.task_id,
                self.task_type,
                success,
                execution_time,
                report_path
            )

            return True
        except Exception as e:
            self.logger.exception(f"Error finalizing task {self.task_id}: {str(e)}")
            return False

    def run(self, args: Optional[Dict[str, Any]] = None) -> bool:
        """
        Run the complete task: initialize, configure, execute, finalize.

        Parameters:
        -----------
        args : Dict[str, Any], optional
            Command line arguments to override configuration

        Returns:
        --------
        bool
            True if the task executed successfully, False otherwise
        """
        try:
            # Initialize
            if not self.initialize(args):
                self.logger.error(f"Failed to initialize task: {self.task_id}")
                self.finalize(False)
                return False

            # Configure operations
            self.configure_operations()

            # Execute task
            success = self.execute()

            # Finalize
            self.finalize(success)

            return success

        except Exception as e:
            error_msg = f"Unhandled error in task {self.task_id}: {str(e)}"

            if self.logger:
                self.logger.exception(error_msg)
            else:
                logging.exception(error_msg)

            # Try to finalize with error status
            try:
                self.finalize(False)
            except:
                pass

            return False

    def add_operation(self, operation_class: str, scope: Optional[OperationScope] = None, **kwargs) -> bool:
        """
        Add an operation to the task.

        Parameters:
        -----------
        operation_class : str
            Name of the operation class to instantiate
        scope : OperationScope, optional
            Scope for the operation to work with (fields, datasets, etc.)
        **kwargs : dict
            Parameters for the operation

        Returns:
        --------
        bool
            True if operation was added successfully, False otherwise
        """
        try:
            # Apply encryption settings if not explicitly provided
            if 'use_encryption' not in kwargs and self.use_encryption:
                kwargs['use_encryption'] = self.use_encryption

            if 'encryption_key' not in kwargs and self.encryption_key:
                kwargs['encryption_key'] = self.encryption_key

            # Apply vectorization settings if not explicitly provided
            if 'use_vectorization' not in kwargs and self.use_vectorization:
                kwargs['use_vectorization'] = self.use_vectorization

            # Apply scope settings
            if scope:
                kwargs['scope'] = scope
            elif self.global_scope and 'scope' not in kwargs:
                kwargs['scope'] = self.global_scope

            operation = create_operation_instance(operation_class, **kwargs)

            if operation:
                self.operations.append(operation)
                return True
            else:
                if self.logger:
                    self.logger.error(f"Failed to create operation: {operation_class}")
                return False

        except Exception as e:
            if self.logger:
                self.logger.exception(f"Error adding operation {operation_class}: {str(e)}")
            return False

    def create_operation_scope(self, fields: List[str] = None,
                               datasets: List[str] = None,
                               field_groups: Dict[str, List[str]] = None) -> OperationScope:
        """
        Create an operation scope for use with operations.

        Parameters:
        -----------
        fields : List[str], optional
            List of field names to include in the scope
        datasets : List[str], optional
            List of dataset names to include in the scope
        field_groups : Dict[str, List[str]], optional
            Named groups of fields to include in the scope

        Returns:
        --------
        OperationScope
            A configured operation scope
        """
        return OperationScope(fields=fields, datasets=datasets, field_groups=field_groups)