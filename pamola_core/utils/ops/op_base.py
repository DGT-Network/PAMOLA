"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Base Classes
Description: Base operation classes for defining modular privacy-enhancing tasks
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides base classes for defining operations that can be used
across different domains like profiling, anonymization, security, etc.

Key features:
- Standardized operation lifecycle (initialization, execution, result collection)
- Progress tracking with hierarchical reporting
- Configuration management and serialization
- Atomic artifact management
- Integrated logging and error handling
- Support for encryption of outputs
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import pandas as pd

from pamola_core.utils.ops.op_data_source import DataSource
# Import OperationResult directly to ensure type checker recognizes the class and its methods
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_registry import register_operation

# Configure logger
logger = logging.getLogger(__name__)


class ConfigSaveError(Exception):
    """
    Error raised when saving operation configuration fails.
    """
    pass


class OperationScope:
    """
    Defines the scope of an operation.

    This class allows operations to define which fields, datasets, or data groups
    they should operate on, providing flexibility for different types of operations.
    """

    def __init__(self,
                 fields: Optional[List[str]] = None,
                 datasets: Optional[List[str]] = None,
                 field_groups: Optional[Dict[str, List[str]]] = None):
        """
        Initialize an operation scope.

        Parameters:
        -----------
        fields : List[str], optional
            List of field names to operate on
        datasets : List[str], optional
            List of dataset names to operate on
        field_groups : Dict[str, List[str]], optional
            Named groups of fields to operate on
        """
        self.fields = fields or []
        self.datasets = datasets or []
        self.field_groups = field_groups or {}

    def add_field(self, field_name: str) -> None:
        """Add a field to the scope."""
        if field_name not in self.fields:
            self.fields.append(field_name)

    def add_dataset(self, dataset_name: str) -> None:
        """Add a dataset to the scope."""
        if dataset_name not in self.datasets:
            self.datasets.append(dataset_name)

    def add_field_group(self, group_name: str, fields: List[str]) -> None:
        """Add a named group of fields."""
        self.field_groups[group_name] = fields

    def has_field(self, field_name: str) -> bool:
        """Check if a field is in the scope."""
        return field_name in self.fields

    def has_dataset(self, dataset_name: str) -> bool:
        """Check if a dataset is in the scope."""
        return dataset_name in self.datasets

    def has_field_group(self, group_name: str) -> bool:
        """Check if a field group exists."""
        return group_name in self.field_groups

    def get_fields_in_group(self, group_name: str) -> List[str]:
        """Get the fields in a specific group."""
        return self.field_groups.get(group_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fields": self.fields,
            "datasets": self.datasets,
            "field_groups": self.field_groups
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationScope':
        """Create an OperationScope from a dictionary."""
        return cls(
            fields=data.get("fields"),
            datasets=data.get("datasets"),
            field_groups=data.get("field_groups")
        )


class BaseOperation(ABC):
    """
    Base class for all operations.

    This class defines the interface for operations and provides common
    functionality like logging, progress tracking, and result handling.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 scope: Optional[OperationScope] = None,
                 config: Optional[OperationConfig] = None,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_vectorization: bool = False,
                 encryption_mode: Optional[str] = None):
        """
        Initialize the operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of what the operation does
        scope : OperationScope, optional
            The scope of the operation (fields, datasets, etc.)
        config : OperationConfig, optional
            Configuration parameters for the operation
        use_encryption : bool
            Whether to encrypt output files
        encryption_key : str or Path, optional
            The encryption key or path to a key file
        use_vectorization : bool
            Whether to use vectorized processing
        """
        self.name = name
        self.description = description
        self.scope = scope or OperationScope()
        self.config = config or OperationConfig()
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.encryption_mode = encryption_mode
        self.use_vectorization = use_vectorization
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Operation version information (to be set by subclasses)
        self.version = "1.0.0"  # Semantic versioning

        # Internal state
        self._execution_time = None

        # Register operation in the registry
        register_operation(self.__class__)

    def save_config(self, task_dir: Path) -> None:
        """
        Serialize this operation's config to JSON.

        Writes the configuration to {task_dir}/config.json atomically,
        including operation name and version information.

        Parameters:
        -----------
        task_dir : Path
            Directory where the configuration file should be saved

        Raises:
        -------
        ConfigSaveError
            If the configuration cannot be saved

        Satisfies:
        ----------
        REQ-OPS-004: BaseOperation.save_config(task_dir) writes config.json
                    atomically before execution begins.
        """
        # Create configuration dictionary with operation metadata
        config_dict = self.config.to_dict()
        config_dict.update({
            "operation_name": self.name,
            "version": self.version
        })

        # Ensure task directory exists
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary filename for atomic write
        config_path = task_dir / "config.json"
        temp_path = config_path.with_suffix(".json.tmp")

        try:
            # Write to temporary file first
            with open(temp_path, 'w') as f:
                json.dump(config_dict, f, indent=2) # type: ignore

            # Atomic replace
            os.replace(temp_path, config_path)
            self.logger.info(f"Saved operation configuration to {Path(config_path).name}")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise ConfigSaveError(f"Failed to save configuration: {str(e)}") from e

    @abstractmethod
    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[HierarchicalProgressTracker] = None,
                **kwargs) -> OperationResult:
        """
        Execute the operation.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation (dataframes or file paths)
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for the operation
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation including status, artifacts, and metrics
        """
        pass

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare standard directories for storing operation artifacts.

        Parameters:
        -----------
        task_dir : Path
            Base directory for the task

        Returns:
        --------
        Dict[str, Path]
            Dictionary with standard directory paths
        """
        # Create standard directories
        directories = {
            "output": task_dir / "output",
            "dictionaries": task_dir / "dictionaries",
            "visualizations": task_dir / "visualizations",
            "logs": task_dir / "logs"  # Changed to be inside task_dir
        }

        # Ensure directories exist
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return directories

    def _log_operation_start(self, **kwargs) -> None:
        """
        Log the start of an operation with parameters.

        Logs operation name and all parameters, except sensitive ones.
        """
        self.logger.info(f"Starting operation: {self.name}")

        # List of sensitive parameters that should not be logged
        sensitive_params = ['encryption_key', 'password', 'token', 'secret', 'api_key']

        # Log parameters, skipping sensitive ones
        for key, value in kwargs.items():
            if key in sensitive_params:
                self.logger.debug(f"Parameter {key}: [REDACTED]")
            else:
                # Truncate long values for readability
                if isinstance(value, str) and len(value) > 100:
                    self.logger.debug(f"Parameter {key}: {value[:100]}... [truncated]")
                else:
                    self.logger.debug(f"Parameter {key}: {value}")

    def _log_operation_end(self, result: OperationResult) -> None:
        """
        Log the end of an operation with results.

        Safely handles missing attributes in result objects.
        """
        if not hasattr(result, 'status'):
            self.logger.error(f"Invalid OperationResult object: missing status attribute")
            return

        self.logger.info(f"Operation {self.name} completed with status: {result.status.name}")

        # Safely get execution time
        execution_time = getattr(result, 'execution_time', None)
        if execution_time is not None:
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")

        # Safely get error message
        if result.status == OperationStatus.ERROR:
            error_message = getattr(result, 'error_message', 'Unknown error')
            self.logger.error(f"Error: {error_message}")

    def run(
            self,
            *,  # Force keyword-only arguments for clarity
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            progress_tracker: Optional[HierarchicalProgressTracker] = None,
            track_progress: bool = True,
            parallel_processes: int = 1,
            **kwargs
    ) -> OperationResult:
        """
        Run the operation with timing and error handling.

        This is a wrapper around execute() that adds timing, error handling,
        and progress tracking.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation
        task_dir : Path
            Directory where task artifacts should be saved
        reporter : Any
            Reporter object for tracking progress and artifacts
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for this operation
        track_progress : bool
            Whether to track and report progress
        parallel_processes : int
            Number of parallel processes to use (if operation supports vectorization)
        **kwargs : dict
            Additional parameters for the operation

        Returns:
        --------
        OperationResult
            Results of the operation
        """
        # Handle potential duplicate progress_tracker in kwargs
        # This prevents the "got multiple values for keyword argument" error
        if "progress_tracker" in kwargs:
            # If progress_tracker parameter is None, use the one from kwargs
            if progress_tracker is None:
                progress_tracker = kwargs.pop("progress_tracker")
            else:
                # Otherwise, remove the duplicate from kwargs to avoid TypeError
                kwargs.pop("progress_tracker")

        # Check if encryption is requested but key is not provided
        if self.use_encryption and not self.encryption_key:
            self.logger.warning("Encryption requested but no key provided, disabling encryption")
            self.use_encryption = False

        # Check if vectorization is requested but operation doesn't support it
        if self.use_vectorization and not parallel_processes > 1:
            self.logger.warning("Vectorization requested but parallel_processes <= 1, disabling vectorization")
            self.use_vectorization = False

        # Save configuration
        try:
            self.save_config(task_dir)
        except ConfigSaveError as e:
            self.logger.error(f"Failed to save operation configuration: {str(e)}")
            # Continue execution despite config save failure

        # Log operation start
        self._log_operation_start(**kwargs)

        # Create progress tracker if requested and not already provided
        if track_progress and progress_tracker is None:
            total_steps = kwargs.get('total_steps', 3)
            progress_tracker = HierarchicalProgressTracker(
                total=total_steps,
                description=f"Operation: {self.name}",
                unit="steps"
            )

        # Add operation to reporter
        if reporter:
            reporter.add_operation(self.name, details={
                "description": self.description,
                "parameters": {k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))},
                "use_encryption": self.use_encryption,
                "use_vectorization": self.use_vectorization,
                "version": self.version,
                "scope": self.scope.to_dict() if hasattr(self.scope, 'to_dict') else None
            })

        # Record start time
        start_time = time.time()

        try:
            # Setup encryption params for I/O operations if needed
            io_params = {}
            if self.use_encryption:
                io_params['use_encryption'] = True
                io_params['encryption_key'] = self.encryption_key

            # Setup vectorization if requested
            vectorization_params = {}
            if self.use_vectorization:
                vectorization_params['parallel_processes'] = parallel_processes

            # Combine all params for execution
            execution_params = {**kwargs, **io_params, **vectorization_params}

            # Execute the operation with the correct progress_tracker
            result: OperationResult = self.execute(
                data_source=data_source,
                task_dir=task_dir,
                reporter=reporter,
                progress_tracker=progress_tracker,
                **execution_params
            )

            # Set execution time
            result.execution_time = time.time() - start_time
            self._execution_time = result.execution_time

            # Add final operation status to reporter
            if reporter:
                # Create standardized details dictionary
                details_dict = {
                    "status": result.status.value,
                    "execution_time": f"{result.execution_time:.2f} seconds" if result.execution_time else None,
                }

                # Add error message if present
                if hasattr(result, 'error_message') and result.error_message:
                    details_dict["error_message"] = result.error_message

                # Add additional details from result if available
                if hasattr(result, 'to_reporter_details') and callable(getattr(result, 'to_reporter_details')):
                    additional_details = result.to_reporter_details()
                    if additional_details:
                        details_dict.update(additional_details)

                reporter.add_operation(
                    f"{self.name} completed",
                    status="success" if result.status == OperationStatus.SUCCESS else "warning",
                    details=details_dict
                )

        except Exception as e:
            # Handle exceptions
            self.logger.exception(f"Error in operation {self.name}: {str(e)}")

            # Close progress tracker if one was created
            if progress_tracker:
                progress_tracker.close()

            # Create error result
            result = OperationResult(
                status=OperationStatus.ERROR,
                error_message=str(e),
                execution_time=time.time() - start_time
            )

            # Add error to reporter
            if reporter:
                reporter.add_operation(
                    f"{self.name} failed",
                    status="error",
                    details={"error": str(e)}
                )

        # Log operation end
        self._log_operation_end(result)

        return result

    def get_execution_time(self) -> Optional[float]:
        """Get the execution time of the last run."""
        return self._execution_time

    def get_version(self) -> str:
        """Get the version of the operation."""
        return self.version

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        # Nothing to clean up in the base class
        return False  # Don't suppress exceptions


class FieldOperation(BaseOperation):
    """
    Base class for operations that process specific fields.
    """

    def __init__(self,
                 field_name: str,
                 description: str = "",
                 config: Optional[OperationConfig] = None,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_vectorization: bool = False,
                 encryption_mode: Optional[str] = None):
        """
        Initialize a field-specific operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        description : str
            Description of what the operation does
        config : OperationConfig, optional
            Configuration parameters for the operation
        use_encryption : bool
            Whether to encrypt output files
        encryption_key : str or Path, optional
            The encryption key or path to a key file
        use_vectorization : bool
            Whether to use vectorized processing
        """
        # Create a scope with the specified field
        scope = OperationScope(fields=[field_name])

        super().__init__(
            name=f"{field_name} analysis",
            description=description or f"Analysis of {field_name} field",
            scope=scope,
            config=config,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            use_vectorization=use_vectorization,
            encryption_mode=encryption_mode
        )
        self.field_name = field_name

    def add_related_field(self, field_name: str) -> None:
        """
        Add a related field to the operation's scope.

        Parameters:
        -----------
        field_name : str
            Name of the related field
        """
        self.scope.add_field(field_name)

    def validate_field_existence(self, df: pd.DataFrame) -> bool:
        """
        Validate that the main field exists in the DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to check

        Returns:
        --------
        bool
            True if field exists, False otherwise
        """
        if self.field_name not in df.columns:
            self.logger.error(f"Field {self.field_name} not found in DataFrame")
            return False
        return True


class DataFrameOperation(BaseOperation):
    """
    Base class for operations that process entire DataFrames.
    """

    def __init__(self,
                 name: str,
                 description: str = "",
                 scope: Optional[OperationScope] = None,
                 config: Optional[OperationConfig] = None,
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_vectorization: bool = False):
        """
        Initialize a DataFrame operation.

        Parameters:
        -----------
        name : str
            Name of the operation
        description : str
            Description of what the operation does
        scope : OperationScope, optional
            The scope of the operation (datasets, field groups, etc.)
        config : OperationConfig, optional
            Configuration parameters for the operation
        use_encryption : bool
            Whether to encrypt output files
        encryption_key : str or Path, optional
            The encryption key or path to a key file
        use_vectorization : bool
            Whether to use vectorized processing
        """
        super().__init__(
            name=name,
            description=description,
            scope=scope or OperationScope(),
            config=config,
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            use_vectorization=use_vectorization
        )

    def add_field_group(self, group_name: str, fields: List[str]) -> None:
        """
        Add a group of fields to process together.

        Parameters:
        -----------
        group_name : str
            Name of the field group
        fields : List[str]
            List of field names in the group
        """
        self.scope.add_field_group(group_name, fields)

    def get_field_group(self, group_name: str) -> List[str]:
        """
        Get the fields in a specific group.

        Parameters:
        -----------
        group_name : str
            Name of the field group

        Returns:
        --------
        List[str]
            List of field names in the group
        """
        return self.scope.get_fields_in_group(group_name)