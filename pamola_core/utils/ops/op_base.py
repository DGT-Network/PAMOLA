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
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union, Optional

from pamola_core.common.type_aliases import DataFrameType
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import HierarchicalProgressTracker
from pamola_core.utils import logging

# Configure module logger
logger = logging.get_logger(__name__)


class ConfigSaveError(Exception):
    """Error raised when saving operation configuration fails."""

    pass


class OperationScope:
    """
    Defines the scope of an operation.

    This class allows operations to define which datasets, fields, or data groups
    they should operate on, providing flexibility for different types of operations.
    """

    def __init__(
        self,
        datasets: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        field_groups: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Initialize an operation scope.

        Parameters:
        -----------
        datasets : List[str], optional
            List of dataset names to operate on
        fields : List[str], optional
            List of field names to operate on
        field_groups : Dict[str, List[str]], optional
            Named groups of fields to operate on
        """
        self.datasets = datasets or []
        self.fields = fields or []
        self.field_groups = field_groups or {}

    def add_dataset(self, dataset_name: str) -> None:
        """Add a dataset to the scope."""
        if dataset_name not in self.datasets:
            self.datasets.append(dataset_name)

    def add_field(self, field_name: str) -> None:
        """Add a field to the scope."""
        if field_name not in self.fields:
            self.fields.append(field_name)

    def add_field_group(self, group_name: str, fields: List[str]) -> None:
        """Add a named group of fields."""
        self.field_groups[group_name] = fields

    def has_dataset(self, dataset_name: str) -> bool:
        """Check if a dataset is in the scope."""
        return dataset_name in self.datasets

    def has_field(self, field_name: str) -> bool:
        """Check if a field is in the scope."""
        return field_name in self.fields

    def has_field_group(self, group_name: str) -> bool:
        """Check if a field group exists."""
        return group_name in self.field_groups

    def get_fields_in_group(self, group_name: str) -> List[str]:
        """Get the fields in a specific group."""
        return self.field_groups.get(group_name, [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "datasets": self.datasets,
            "fields": self.fields,
            "field_groups": self.field_groups,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperationScope":
        """Create an OperationScope from a dictionary."""
        return cls(
            datasets=data.get("datasets"),
            fields=data.get("fields"),
            field_groups=data.get("field_groups"),
        )


class BaseOperation(ABC):
    """
    Base class for all operations.

    This class defines the interface for all operation types and provides
    common functionality such as configuration management, logging,
    performance tracking, and output control.
    """

    def __init__(
        self,
        # Identification & Meta
        name: str = "",
        description: str = "",
        scope: Optional[OperationScope] = None,
        config: Optional[OperationConfig] = None,

        # Pre-processing & Performance
        optimize_memory: bool = True,
        adaptive_chunk_size: bool = True,

        # Processing control
        mode: str = "REPLACE",
        column_prefix: str = "_",
        output_field_name: Optional[str] = None,
        null_strategy: str = "PRESERVE",
        engine: str = "auto",
        use_dask: bool = False,
        npartitions: Optional[int] = None,
        dask_partition_size: Optional[str] = None,
        use_vectorization: bool = False,
        parallel_processes: Optional[int] = None,
        chunk_size: int = 10000,

        # Output & Result
        use_cache: bool = False,
        output_format: str = "csv",
        visualization_theme: Optional[str] = None,
        visualization_backend: Optional[str] = "plotly",
        visualization_strict: bool = False,
        visualization_timeout: int = 120,

        # Security & Encryption
        use_encryption: bool = False,
        encryption_mode: Optional[str] = None,
        encryption_key: Optional[Union[str, Path]] = None,

        # Runtime control flags
        force_recalculation: bool = False,
        generate_visualization: bool = True,
        save_output: bool = True,
    ):
        """
        Initialize a new BaseOperation instance.

        Parameters
        ----------
        name : str, optional
            Name of the operation (default is an empty string).
        description : str, optional
            Description of the operation (default is empty).
        scope : OperationScope, optional
            Defines the operational scope of this operation.
        config : OperationConfig, optional
            Configuration schema instance (default is None).

        optimize_memory : bool, optional
            Whether to enable memory optimization techniques (default=True).
        adaptive_chunk_size : bool, optional
            Adjust chunk size dynamically based on available memory (default=True).

        mode : {"REPLACE", "ENRICH"}, optional
            Processing mode for the operation (default="REPLACE").
        column_prefix : str, optional
            Prefix for newly generated columns (default="_").
        output_field_name : str, optional
            Configuration schema instance (default is None).

        null_strategy : {"PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"}, optional
            Strategy for handling null values (default="PRESERVE").
        engine : {"pandas", "dask", "auto"}, optional
            Processing engine to use (default="auto").
        use_dask : bool, optional
            Whether to enable Dask-based parallel execution (default=False).
        npartitions : int, optional
            Number of Dask partitions (default=None).
        dask_partition_size : str, optional
            Target partition size (e.g., "128MB", default=None).
        use_vectorization : bool, optional
            Whether to enable vectorized computation (default=False).
        parallel_processes : int, optional
            Number of parallel worker processes (default=None).
        chunk_size : int, optional
            Data chunk size per iteration (default=10000).

        use_cache : bool, optional
            Enable caching for repeated runs (default=False).
        output_format : {"csv", "json", "parquet"}, optional
            Output format for saved results (default="csv").
        visualization_theme : str, optional
            Visualization theme to use (default=None).
        visualization_backend : {"plotly", "matplotlib"}, optional
            Backend for rendering visualizations (default="plotly").
        visualization_strict : bool, optional
            Raise exceptions on visualization errors if True (default=False).
        visualization_timeout : int, optional
            Timeout in seconds for visualization generation (default=120).

        use_encryption : bool, optional
            Enable encryption for stored results (default=False).
        encryption_mode : str, optional
            Encryption mode or algorithm (default=None).
        encryption_key : str or Path, optional
            Encryption key or path to key file (default=None).

        force_recalculation : bool, optional
            Force recomputation even if cached results exist (default=False).
        generate_visualization : bool, optional
            Automatically generate visualization after processing (default=True).
        save_output : bool, optional
            Save processed output to disk (default=True)
        """

        # Identification & meta
        self.name = name
        self.description = description
        self.scope = scope or OperationScope()
        self.config = config or OperationConfig()

        # Pre-processing & performance
        self.optimize_memory = optimize_memory
        self.adaptive_chunk_size = adaptive_chunk_size
        self.original_chunk_size = chunk_size

        # Processing control
        self.mode = mode.upper()
        self.column_prefix = column_prefix
        self.output_field_name = output_field_name
        self.null_strategy = null_strategy.upper()
        self.engine = engine.lower()
        self.use_dask = use_dask
        self.npartitions = npartitions
        self.dask_partition_size = dask_partition_size
        self.use_vectorization = use_vectorization
        self.parallel_processes = parallel_processes
        self.chunk_size = chunk_size

        # Output & visualization
        self.use_cache = use_cache
        self.output_format = output_format
        self.visualization_theme = visualization_theme
        self.visualization_backend = visualization_backend
        self.visualization_strict = visualization_strict
        self.visualization_timeout = visualization_timeout

        # Security & encryption
        self.use_encryption = use_encryption
        self.encryption_mode = encryption_mode
        self.encryption_key = encryption_key

        # Runtime control
        self.force_recalculation = force_recalculation
        self.generate_visualization = generate_visualization
        self.save_output = save_output

        # Internal runtime state
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.version = "1.0.0"
        self.operation_name = self.__class__.__name__
        self.operation_cache = None

        # Performance tracking
        self.process_count: int = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.execution_time: Optional[float] = None

        # Register this operation in the registry
        register_operation(operation_class=self.__class__)

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
        config_dict.update({"operation_name": self.name, "version": self.version})

        # Ensure task directory exists
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary filename for atomic write
        config_path = task_dir / "config.json"
        temp_path = config_path.with_suffix(".json.tmp")

        try:
            # Write to temporary file first
            with open(temp_path, "w") as f:
                json.dump(config_dict, f, indent=2)  # type: ignore

            # Atomic replace
            os.replace(temp_path, config_path)
            self.logger.info(
                f"Saved operation configuration to {Path(config_path).name}"
            )
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {str(e)}")
            raise ConfigSaveError(f"Failed to save configuration: {str(e)}") from e

    @abstractmethod
    def execute(
        self,
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        **kwargs,
    ) -> OperationResult:
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

    def _check_dask_availability(self) -> bool:
        """
        Check if Dask is available for use.

        Returns:
        --------
        bool
            True if Dask is available, False otherwise
        """
        try:
            import dask.dataframe as dd

            return True
        except ImportError:
            if self.engine == "dask":
                self.logger.error(
                    "Dask explicitly requested but not available. Install with: pip install dask[complete]"
                )
            return False

    def _should_use_dask(
        self, data_source: DataSource, dataset_name: str = "main"
    ) -> bool:
        """
        Determine whether to use Dask based on data size and configuration.

        Parameters:
        -----------
        data_source : DataSource
            Source of data for the operation (dataframes or file paths)
        dataset_name : str, optional
            Name of the DataFrame

        Returns:
        --------
        bool
            True if Dask should be used, False otherwise
        """
        if not self._check_dask_availability():
            return False

        if self.engine == "dask":
            return True
        elif self.engine == "pandas":
            return False
        else:  # auto
            engine = data_source.suggest_engine(name=dataset_name)
            return engine == "dask"

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts following PAMOLA.CORE conventions.

        Parameters:
        -----------
        task_dir : Path
            Root task directory

        Returns:
        --------
        Dict[str, Path]
            Dictionary with prepared directories
        """
        # Create standard directories following PAMOLA.CORE conventions
        directories = {
            "root": task_dir,
            "output": task_dir / "output",
            "dictionaries": task_dir / "dictionaries",
            "visualizations": task_dir / "visualizations",
            "metrics": task_dir / "metrics",
            "cache": task_dir / "cache",
            "logs": task_dir / "logs",
            "reports": task_dir / "reports",
        }

        # Ensure all directories exist
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
        sensitive_params = ["encryption_key", "password", "token", "secret", "api_key"]

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
        if not hasattr(result, "status"):
            self.logger.error(
                f"Invalid OperationResult object: missing status attribute"
            )
            return

        self.logger.info(
            f"Operation {self.name} completed with status: {result.status.name}"
        )

        # Safely get execution time
        execution_time = getattr(result, "execution_time", None)
        if execution_time is not None:
            self.logger.info(f"Execution time: {execution_time:.2f} seconds")

        # Safely get error message
        if result.status == OperationStatus.ERROR:
            error_message = getattr(result, "error_message", "Unknown error")
            self.logger.error(f"Error: {error_message}")

    def run(
        self,
        *,  # Force keyword-only arguments for clarity
        data_source: DataSource,
        task_dir: Path,
        reporter: Any,
        progress_tracker: Optional[HierarchicalProgressTracker] = None,
        track_progress: bool = True,
        **kwargs,
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
            self.logger.warning(
                "Encryption requested but no key provided, disabling encryption"
            )
            self.use_encryption = False

        # Check if vectorization is requested but operation doesn't support it
        validate_parallel_processes = (
            self.parallel_processes is None
            or self.parallel_processes == -1
            or self.parallel_processes > 1
        )
        if self.use_vectorization and not validate_parallel_processes:
            self.logger.warning(
                "Vectorization requested but parallel_processes not validated, disabling vectorization"
            )
            self.use_vectorization = False

        # Check if dask is requested but operation doesn't support it
        if self.use_dask and not self._should_use_dask(
            data_source, kwargs.get("dataset_name", "main")
        ):
            self.logger.warning("Dask requested but not available, disabling Dask")
            self.use_dask = False

        # Save configuration
        try:
            self.save_config(task_dir=task_dir)
        except ConfigSaveError as e:
            self.logger.error(f"Failed to save operation configuration: {str(e)}")
            # Continue execution despite config save failure

        # Log operation start
        self._log_operation_start(**kwargs)

        # Create progress tracker if requested and not already provided
        if track_progress and progress_tracker is None:
            total_steps = kwargs.get("total_steps", 3)
            progress_tracker = HierarchicalProgressTracker(
                description=f"Operation: {self.name}", total=total_steps, unit="steps"
            )

        # Add operation to reporter
        if reporter:
            reporter.add_operation(
                name=self.name,
                details={
                    "description": self.description,
                    "scope": (
                        self.scope.to_dict() if hasattr(self.scope, "to_dict") else None
                    ),
                    "parameters": {
                        k: v
                        for k, v in kwargs.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                    "use_dask": self.use_dask,
                    "use_vectorization": self.use_vectorization,
                    "use_encryption": self.use_encryption,
                    "use_cache": self.use_cache,
                    "version": self.version,
                },
            )

        # Record start time
        start_time = time.time()

        try:
            # Setup pre-process config
            pre_process_config = {
                "optimize_memory": self.optimize_memory,
                "adaptive_chunk_size": self.adaptive_chunk_size,
            }

            # Setup process config
            process_config = {
                "mode": self.mode,
                "column_prefix": self.column_prefix,
                "null_strategy": self.null_strategy,
                "engine": self.engine,
                "use_dask": self.use_dask,
                "npartitions": self.npartitions,
                "dask_partition_size": self.dask_partition_size,
                "use_vectorization": self.use_vectorization,
                "parallel_processes": self.parallel_processes,
                "chunk_size": self.chunk_size,
            }

            # Setup output config
            output_config = {
                "use_cache": self.use_cache,
                "output_format": self.output_format,
                "visualization_theme": self.visualization_theme,
                "visualization_backend": self.visualization_backend,
                "visualization_strict": self.visualization_strict,
                "visualization_timeout": self.visualization_timeout,
                "use_encryption": self.use_encryption,
                "encryption_mode": self.encryption_mode,
                "encryption_key": self.encryption_key,
            }

            # Combine all params for execution
            execution_params = {
                **pre_process_config,
                **process_config,
                **output_config,
                **kwargs,
            }

            # Execute the operation with the correct progress_tracker
            result: OperationResult = self.execute(
                data_source=data_source,
                task_dir=task_dir,
                reporter=reporter,
                progress_tracker=progress_tracker,
                **execution_params,
            )

            # Set execution time
            result.execution_time = time.time() - start_time
            self.execution_time = result.execution_time

            # Add final operation status to reporter
            if reporter:
                # Create standardized details dictionary
                details_dict = {
                    "status": result.status.value,
                    "execution_time": (
                        f"{result.execution_time:.2f} seconds"
                        if result.execution_time
                        else None
                    ),
                }

                # Add error message if present
                if hasattr(result, "error_message") and result.error_message:
                    details_dict["error_message"] = result.error_message

                # Add additional details from result if available
                if hasattr(result, "to_reporter_details") and callable(
                    getattr(result, "to_reporter_details")
                ):
                    additional_details = result.to_reporter_details()
                    if additional_details:
                        details_dict.update(additional_details)

                reporter.add_operation(
                    name=f"{self.name} completed",
                    status=(
                        "success"
                        if result.status == OperationStatus.SUCCESS
                        else "warning"
                    ),
                    details=details_dict,
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
                execution_time=time.time() - start_time,
                exception=e,
            )

            # Add error to reporter
            if reporter:
                reporter.add_operation(
                    name=f"{self.name} failed",
                    status="error",
                    details={"error": str(e)},
                )

        # Log operation end
        self._log_operation_end(result)

        return result

    def get_execution_time(self) -> Optional[float]:
        """Get the execution time of the last run."""
        return self.execution_time

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


class FieldOperation(BaseOperation, ABC):
    """
    Base class for operations that process specific fields.

    Extends :class:`BaseOperation` to handle per-field logic while inheriting
    configuration, performance, and output settings from the base class.
    """

    def __init__(
        self,
        field_name: str,
        **kwargs
    ):
        """
        Initialize a field-specific operation.

        Parameters
        ----------
        field_name : str
            Name of the field (column) to process.
        output_field_name : str, optional
            Optional name for the output field.
        **kwargs : dict
            Additional keyword arguments passed to :class:`BaseOperation.__init__`.

        Notes
        -----
        - Automatically constructs an :class:`OperationScope` for this field.
        - Inherits all configuration and runtime parameters from :class:`BaseOperation`.
        """
        # Auto-create a scope for the given field
        scope = OperationScope(fields=[field_name])
        kwargs["scope"] = scope  # Inject into kwargs before calling super()

        # Optionally inject defaults for name/description if not given
        kwargs.setdefault("name", f"{field_name}_operation")
        kwargs.setdefault("description", f"Operation applied to '{field_name}'")

        # Initialize the base operation
        super().__init__(**kwargs)

        # Field-specific attributes
        self.field_name = field_name
        self.output_field_name = self.output_field_name or (
            field_name
            if self.mode == "REPLACE"
            else f"{self.column_prefix}{field_name}"
        )

        # Debug info
        self.logger.debug(
            f"Initialized FieldOperation for field '{self.field_name}' "
            f"-> output '{self.output_field_name}' | mode={self.mode}"
        )

    def add_related_field(self, field_name: str) -> None:
        """
        Add a related field to the operation's scope.

        Parameters:
        -----------
        field_name : str
            Name of the related field
        """
        self.scope.add_field(field_name)

    def validate_field_existence(self, df: DataFrameType) -> bool:
        """
        Validate that the main field exists in the DataFrame.

        Parameters:
        -----------
        df : DataFrameType
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


class DataFrameOperation(BaseOperation, ABC):
    """
    Base class for operations applied to an entire DataFrame.

    Extends :class:`BaseOperation` to handle DataFrame-wide transformations
    while maintaining consistent lifecycle, configuration, and scope behavior.
    """

    def __init__(self, **kwargs):
        """
        Initialize a DataFrame operation.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments passed to :class:`BaseOperation.__init__`.

        Notes
        -----
        - Default `name` and `description` will be inferred if not provided.
        """
        # Inject default name, description if not provided
        kwargs.setdefault("name", "dataframe_operation")
        kwargs.setdefault("description", "Operation applied to the entire DataFrame")

        # Initialize the base operation
        super().__init__(**kwargs)

        # Debug information
        self.logger.debug(
            f"Initialized DataFrameOperation: {self.name} | scope={self.scope} | mode={self.mode}"
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
