"""
Base classes for task operations in the HHR project.

This module provides base classes for defining operations that can be used
across different domains like profiling, anonymization, security, etc.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

import pandas as pd

from pamola_core.utils.ops.op_data_source import DataSource
# Import OperationResult directly to ensure type checker recognizes the class and its methods
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker

# Configure logger
logger = logging.getLogger(__name__)


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

    def add_field(self, field_name: str):
        """Add a field to the scope."""
        if field_name not in self.fields:
            self.fields.append(field_name)

    def add_dataset(self, dataset_name: str):
        """Add a dataset to the scope."""
        if dataset_name not in self.datasets:
            self.datasets.append(dataset_name)

    def add_field_group(self, group_name: str, fields: List[str]):
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
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_vectorization: bool = False):
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
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.use_vectorization = use_vectorization
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Operation version information (to be set by subclasses)
        self.version = "1.0.0"  # Semantic versioning

        # Internal state
        self._execution_time = None

    @abstractmethod
    def execute(self,
                data_source: DataSource,
                task_dir: Path,
                reporter: Any,
                progress_tracker: Optional[ProgressTracker] = None,
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
        progress_tracker : ProgressTracker, optional
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
            "logs": task_dir.parent / "logs"
        }

        # Ensure directories exist
        for dir_path in directories.values():
            dir_path.mkdir(parents=True, exist_ok=True)

        return directories

    def _log_operation_start(self, **kwargs):
        """Log the start of an operation with parameters."""
        self.logger.info(f"Starting operation: {self.name}")
        for key, value in kwargs.items():
            self.logger.debug(f"Parameter {key}: {value}")

    def _log_operation_end(self, result: OperationResult):
        """Log the end of an operation with results."""
        self.logger.info(f"Operation {self.name} completed with status: {result.status.name}")
        if result.execution_time:
            self.logger.info(f"Execution time: {result.execution_time:.2f} seconds")
        if result.status == OperationStatus.ERROR:
            self.logger.error(f"Error: {result.error_message}")

    def run(self,
            data_source: DataSource,
            task_dir: Path,
            reporter: Any,
            track_progress: bool = True,
            parallel_processes: int = 1,
            **kwargs) -> OperationResult:
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
        # Check if encryption is requested but key is not provided
        if self.use_encryption and not self.encryption_key:
            self.logger.warning("Encryption requested but no key provided, disabling encryption")
            self.use_encryption = False

        # Check if vectorization is requested but operation doesn't support it
        if self.use_vectorization and not parallel_processes > 1:
            self.logger.warning("Vectorization requested but parallel_processes <= 1, disabling vectorization")
            self.use_vectorization = False

        # Log operation start
        self._log_operation_start(**kwargs)

        # Create progress tracker if requested
        progress_tracker = None
        if track_progress:
            total_steps = kwargs.get('total_steps', 3)
            progress_tracker = ProgressTracker(
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

            # Execute the operation
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
                # Explicitly get details dictionary rather than calling method directly
                details_dict = {}
                if hasattr(result, 'to_reporter_details') and callable(getattr(result, 'to_reporter_details')):
                    details_dict = result.to_reporter_details()
                else:
                    # Fallback if method doesn't exist
                    details_dict = {
                        "status": result.status.value,
                        "execution_time": f"{result.execution_time:.2f} seconds" if result.execution_time else None,
                        "error_message": result.error_message if hasattr(result, 'error_message') else None
                    }

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


class FieldOperation(BaseOperation):
    """
    Base class for operations that process specific fields.
    """

    def __init__(self,
                 field_name: str,
                 description: str = "",
                 use_encryption: bool = False,
                 encryption_key: Optional[Union[str, Path]] = None,
                 use_vectorization: bool = False):
        """
        Initialize a field-specific operation.

        Parameters:
        -----------
        field_name : str
            Name of the field to process
        description : str
            Description of what the operation does
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
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            use_vectorization=use_vectorization
        )
        self.field_name = field_name

    def add_related_field(self, field_name: str):
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
            use_encryption=use_encryption,
            encryption_key=encryption_key,
            use_vectorization=use_vectorization
        )

    def add_field_group(self, group_name: str, fields: List[str]):
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