"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
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

import functools
import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Union, Optional
import pandas as pd
import dask.dataframe as dd

from pamola_core.common.type_aliases import DataFrameType
from pamola_core.utils.helpers import generate_data_hash
from pamola_core.utils.redaction import (
    is_sensitive_parameter,
    redact_sensitive_parameters,
    secret_identity,
)
from pamola_core.utils.ops.op_config import OperationConfig
from pamola_core.utils.ops.op_data_source import DataSource
from pamola_core.utils.ops.op_registry import register_operation
from pamola_core.utils.ops.op_result import (
    OperationResult,
    OperationStatus,
)
from pamola_core.utils.progress import HierarchicalProgressTracker
import pamola_core.utils.logging as pamola_logging
from pamola_core.utils.io import ensure_directory
from pamola_core.errors.codes import ErrorCode
from pamola_core.errors.base import BasePamolaError
from pamola_core.errors.error_handler import ErrorHandler
from pamola_core.errors.exceptions import (
    ConfigSaveError,
    DependencyMissingError,
    OutputFieldExistsError,
)
from pamola_core.utils.optional_deps import require_optional

# Configure module logger
logger = pamola_logging.getLogger(__name__)

# Modules an explicitly requested visualization needs, per rendering backend.
# plotly writes PNGs through kaleido; both ship with the ``viz`` extra.
_VISUALIZATION_BACKEND_MODULES: Dict[str, tuple] = {
    "plotly": ("plotly", "kaleido"),
    "matplotlib": ("matplotlib",),
}


def _guard_visualization_dependencies(execute):
    """Wrap ``execute`` so a missing ``viz`` extra fails once, up front.

    Without the check, an operation with ``generate_visualization=True`` on a
    base install does all its work, then every layer of the plotting stack logs
    its own copy of the failure (about twelve WARNING/ERROR lines with
    tracebacks per call) and the result still says SUCCESS. The caller asked for
    charts and cannot get them, so the operation returns one ERROR result that
    names the extra to install, before any data is touched.
    """

    if getattr(execute, "_viz_dependency_guarded", False):
        return execute

    @functools.wraps(execute)
    def guarded(self, *args, **kwargs):
        try:
            self.check_visualization_dependencies()
        except DependencyMissingError as exc:
            message = str(exc)
            self.logger.error(message)
            return OperationResult(
                status=OperationStatus.ERROR,
                error_message=message,
                exception=exc,
            )
        return execute(self, *args, **kwargs)

    guarded._viz_dependency_guarded = True
    return guarded


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

        Parameters
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

    #: Whether this operation can be driven as a pure in-memory
    #: ``DataFrame -> DataFrame`` transform via ``process_batch()``, rather than
    #: only through the full :meth:`execute` lifecycle with its file I/O.
    #:
    #: Declared here, on the root base class, so that **every** operation
    #: answers the question. A caller embedding this library should be able to
    #: write ``if op.supports_batch:`` without a ``getattr`` default and without
    #: knowing which intermediate base a given operation happens to derive from.
    #:
    #: The default is ``False`` — conservative on purpose. Profiling and metrics
    #: operations summarise data rather than rewrite rows and have no batch form
    #: at all; for them ``False`` is simply true. The transform bases
    #: (``AnonymizationOperation``, ``TransformationOperation``,
    #: ``BaseGeneratorOperation``) override it to ``True``, and the handful of
    #: their subclasses that genuinely cannot be batched set it back to
    #: ``False`` individually.
    supports_batch: ClassVar[bool] = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # One framework-level check instead of one per ``execute`` override.
        execute = cls.__dict__.get("execute")
        if callable(execute):
            cls.execute = _guard_visualization_dependencies(execute)

    def check_visualization_dependencies(self) -> None:
        """Raise ``DependencyMissingError`` if requested visualization cannot run.

        A no-op unless ``generate_visualization`` is true and a backend is set,
        so a default-constructed operation never imports a plotting library.

        Raises
        ------
        DependencyMissingError
            Naming the ``viz`` extra, when the selected backend (or kaleido for
            plotly PNG export) is not installed.
        """
        if not getattr(self, "generate_visualization", False):
            return
        backend = getattr(self, "visualization_backend", None)
        if backend is None:
            return
        for module in _VISUALIZATION_BACKEND_MODULES.get(str(backend).lower(), ()):
            require_optional(module, extra="viz")

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
        overwrite_existing_output: bool = False,
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
        generate_visualization: bool = False,
        save_output: bool = True,
        # Reproducibility
        seed: Optional[int] = None,
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
            Name of the ENRICH output column (default: ``column_prefix`` + field).
        overwrite_existing_output : bool, optional
            ENRICH only. When the output column already exists the operation
            fails with ``OutputFieldExistsError``; set True to replace it
            (default=False).

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
            Generate visualization artifacts during ``execute()`` (default=False).
            Opt-in: it needs the ``viz`` extra and adds rendering time to every
            run. When True the artifacts are produced and registered before
            ``execute()`` returns; if the ``viz`` extra is missing the operation
            returns one actionable dependency error.
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
        self.overwrite_existing_output = overwrite_existing_output
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
        self.seed = seed

        # Internal runtime state
        self.logger = pamola_logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.version = "1.0.0"
        self.operation_name = self.__class__.__name__
        self.operation_cache = None
        self.error_handler = None

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

        Parameters
        -----------
        task_dir : Path
            Directory where the configuration file should be saved

        Raises
        -------
        ConfigSaveError
            If the configuration cannot be saved

        Satisfies
        ----------
        REQ-OPS-004: BaseOperation.save_config(task_dir) writes config.json
                    atomically before execution begins.
        """
        # Create configuration dictionary with operation metadata.
        # Use to_safe_dict() so that sensitive parameters (encryption keys,
        # passwords) declared in OperationConfig.SENSITIVE_KEYS are redacted
        # before being written to disk.
        config_dict = self.config.to_safe_dict()
        config_dict.update({"operation_name": self.name, "version": self.version})

        # Ensure task directory exists
        task_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary filename for atomic write
        config_path = task_dir / "config.json"
        temp_path = config_path.with_suffix(".json.tmp")

        # Sanitize before dump
        config_dict = self.config.json_safe(config_dict)

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

        Parameters
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

        Returns
        --------
        OperationResult
            Results of the operation including status, artifacts, and metrics
        """
        pass

    def _check_dask_availability(self) -> bool:
        """
        Check if Dask is available for use.

        Returns
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

        Parameters
        -----------
        data_source : DataSource
            Source of data for the operation (dataframes or file paths)
        dataset_name : str, optional
            Name of the DataFrame

        Returns
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

    def _record_visualization_failure(self, result: Any, error: BaseException) -> None:
        """Make a failed visualization visible in the result.

        An explicitly requested visualization is part of ``execute()``: it is
        waited for (no timeout, no worker left running) and, when it fails, the
        failure is reported in ``result.metrics["visualization_error"]`` and the
        log instead of vanishing. The operation's own data outputs are unaffected,
        so its status stays as the data processing left it.
        """
        message = f"{type(error).__name__}: {error}"
        self.logger.error(f"Visualization failed: {message}")
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            metrics["visualization_error"] = message

    def get_enrich_targets(self, df: DataFrameType) -> List[str]:
        """Names of the columns this operation writes in ENRICH mode.

        The default covers single-field operations (``field_name`` with
        ``output_field_name`` or ``column_prefix``). An operation that selects its
        fields some other way (a per-field strategy table, a list of fields)
        overrides this and returns every column it will write, so that
        ``_check_enrich_target`` can validate all of them before any work is done.
        """
        field_name = getattr(self, "field_name", None)
        if not field_name:
            return []
        return [self.output_field_name or f"{self.column_prefix}{field_name}"]

    def _prefixed_targets(self, field_names: Any, df: DataFrameType, must_exist: bool = True) -> List[str]:
        """ENRICH targets of an operation that writes ``column_prefix`` + each field.

        An empty prefix is not an exemption: the target is then the source column itself
        (the operation edits it in place), which exists, so the guard reports it.
        """
        prefix = self.column_prefix or ""
        return [
            f"{prefix}{name}"
            for name in (field_names or {})
            if name and (not must_exist or name in df.columns)
        ]

    def _enrich_requested(self) -> bool:
        """Whether this run writes ENRICH columns (operations with their own mode switch
        override this)."""
        return self.mode == "ENRICH"

    def _check_enrich_target(self, df: DataFrameType) -> None:
        """Refuse to silently overwrite an existing ENRICH output column.

        Called where the transform bases load their input, so every operation
        built on them gets the same behaviour: ``OutputFieldExistsError`` unless
        ``overwrite_existing_output`` is set. Every target the operation declares
        through ``get_enrich_targets`` is checked. Only column *names* are reported.
        """
        if not self._enrich_requested() or self.overwrite_existing_output:
            return
        for target in self.get_enrich_targets(df):
            if target in df.columns:
                raise OutputFieldExistsError(target, self.operation_name)

    @staticmethod
    def _own_dataframe(df: DataFrameType) -> DataFrameType:
        """Return a frame this operation may freely modify.

        ``DataSource`` hands out the very object the caller registered, so an
        operation that assigns a column (dtype normalisation, a new field, a
        noisy value) rewrites the caller's data. Operations that transform data
        load it through ``_validate_and_get_dataframe``, which calls this right
        after loading: the operation then works on its own deep copy and the
        caller's frame is never touched.

        The price is one transient copy of the frame per ``execute()``. A shallow
        copy would be cheaper but only shields column rebinding; in-place value
        writes (``.loc``/``.iloc``) still reach the caller's buffers, and
        which one an operation uses is an implementation detail it should not
        have to get right. Dask frames are immutable graphs and pass through.
        """
        if isinstance(df, dd.DataFrame):
            return df
        return df.copy(deep=True)

    def _prepare_directories(self, task_dir: Path) -> Dict[str, Path]:
        """
        Prepare directories for artifacts following PAMOLA.CORE conventions.

        Parameters
        -----------
        task_dir : Path
            Root task directory

        Returns
        --------
        Dict[str, Path]
            Dictionary with prepared directories
        """
        # Create standard directories following PAMOLA.CORE conventions
        directories = {
            "root": task_dir,
            "input": task_dir / "input",
            "output": task_dir / "output",
            "temp": task_dir / "temp",
            "dictionaries": task_dir / "dictionaries",
            "visualizations": task_dir / "visualizations",
            "metrics": task_dir / "metrics",
            "attacks": task_dir / "attacks",
            "cache": task_dir / "cache",
            "logs": task_dir / "logs",
            "reports": task_dir / "reports",
        }

        # Ensure all directories exist
        for dir_path in directories.values():
            ensure_directory(dir_path)

        return directories

    def _log_operation_start(self, **kwargs) -> None:
        """
        Log the start of an operation with parameters.

        Logs operation name and all parameters, except sensitive ones.
        """
        self.logger.info(f"Starting operation: {self.name}")

        # Log parameters, skipping sensitive ones (the config's declared
        # SENSITIVE_KEYS as well as secret-looking names)
        declared = getattr(self.config, "SENSITIVE_KEYS", frozenset())
        for key, value in kwargs.items():
            if key in declared or is_sensitive_parameter(key):
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
                "Invalid OperationResult object: missing status attribute"
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

        Parameters
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

        Returns
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
                    "parameters": redact_sensitive_parameters(
                        {
                            k: v
                            for k, v in kwargs.items()
                            if isinstance(v, (str, int, float, bool))
                        },
                        getattr(self.config, "SENSITIVE_KEYS", ()),
                    ),
                    "use_dask": self.use_dask,
                    "use_vectorization": self.use_vectorization,
                    "use_encryption": self.use_encryption,
                    "use_cache": self.use_cache,
                    "version": self.version,
                },
            )

        # Record start time
        start_time = time.time()

        # Create standardized error result
        self.error_handler = ErrorHandler(
            logger=self.logger,
            operation_name=self.name or self.__class__.__name__,
        )

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

            # Standardize error results that were created without error metadata
            if result.status == OperationStatus.ERROR:
                if not result.metrics.get("error_code"):
                    error_code = ErrorCode.PROCESSING_FAILED
                    if (
                        isinstance(result.exception, BasePamolaError)
                        and result.exception.error_code
                    ):
                        error_code = result.exception.error_code
                    self.error_handler.standardize_result(
                        result=result,
                        error_code=error_code,
                        message=result.error_message
                        or str(result.exception or "Unknown error"),
                        context={
                            "operation": self.name or self.__class__.__name__,
                            "task_dir": str(task_dir),
                        },
                    )

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
            # Handle exceptions through centralized handler
            self.logger.exception(f"Error in operation {self.name}: {str(e)}")

            # Close progress tracker if one was created
            if progress_tracker:
                progress_tracker.close()

            result = self.error_handler.handle_error(
                error=e,
                error_code=ErrorCode.PROCESSING_FAILED,
                context={"operation": self.name, "task_dir": str(task_dir)},
                message_kwargs={
                    "field_name": getattr(self, "field_name", "<unspecified>"),
                    "operation": self.name or self.__class__.__name__,
                    "reason": str(e),
                },
            )
            result.execution_time = time.time() - start_time

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

    def _get_base_parameters(self) -> Dict[str, str]:
        """Get the basic parameters for the cache key generation."""
        # Get basic operation parameters
        parameters = {
            # Identification & meta
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "operation_name": self.operation_name,
            # Pre-processing & performance
            "optimize_memory": self.optimize_memory,
            "adaptive_chunk_size": self.adaptive_chunk_size,
            "original_chunk_size": self.original_chunk_size,
            # Processing control
            "mode": self.mode,
            "column_prefix": self.column_prefix,
            "output_field_name": self.output_field_name,
            "overwrite_existing_output": self.overwrite_existing_output,
            "null_strategy": self.null_strategy,
            "engine": self.engine,
            "use_dask": self.use_dask,
            "npartitions": self.npartitions,
            "dask_partition_size": self.dask_partition_size,
            "use_vectorization": self.use_vectorization,
            "parallel_processes": self.parallel_processes,
            "chunk_size": self.chunk_size,
            # Output & visualization
            "use_cache": self.use_cache,
            "output_format": self.output_format,
            "visualization_theme": self.visualization_theme,
            "visualization_backend": self.visualization_backend,
            "visualization_strict": self.visualization_strict,
            "visualization_timeout": self.visualization_timeout,
            # Security & encryption
            "use_encryption": self.use_encryption,
            "encryption_mode": self.encryption_mode,
            # Identity only: the key itself must never reach the cache metadata.
            "encryption_key_id": secret_identity(self.encryption_key),
            # Runtime control
            "force_recalculation": self.force_recalculation,
            "generate_visualization": self.generate_visualization,
            "save_output": self.save_output,
        }

        # Add operation-specific parameters
        parameters.update(self._get_cache_parameters())

        return parameters

    def _get_cache_parameters(self) -> Dict[str, Any]:
        """
        Get operation-specific parameters for cache key generation.

        This method should be overridden by subclasses to provide
        operation-specific parameters for caching.

        Returns
        --------
        Dict[str, Any]
            Parameters for cache key generation
        """
        # Base implementation returns minimal parameters
        return {}

    def _remember_input_for_cache(self, df: Union[pd.DataFrame, dd.DataFrame]) -> None:
        """Fingerprint the WHOLE input frame once, at load time, for the cache key.

        An operation that returns a frame (or an artifact built from one) depends on every
        column it returns, not only on the field it transforms; keying on the field alone
        served a stale CSV after an untouched column changed. Computed before processing
        mutates the working copy, so the key at lookup and at save time is the same. A frame
        that cannot be fingerprinted faithfully marks the call as uncacheable.
        """
        self._input_digest = None
        if not self.use_cache:
            return
        try:
            self._input_digest = generate_data_hash(df)
        except Exception as exc:  # noqa: BLE001 - any failure means "do not cache"
            self._input_digest = exc
            self.logger.warning(f"Caching disabled for this run: {exc}")

    def _cache_data_hash(self, df: Union[pd.Series, pd.DataFrame, dd.DataFrame]) -> str:
        """Fingerprint of the input for the cache key. Raises when none can be made
        faithfully; callers treat that as "do not cache"."""
        digest = getattr(self, "_input_digest", None)
        if isinstance(digest, BaseException):
            raise digest
        if digest is not None:
            return digest
        return generate_data_hash(df)

    def _generate_cache_key(
        self, df: Union[pd.Series, pd.DataFrame, dd.DataFrame]
    ) -> str:
        """
        Generate a deterministic cache key based on operation parameters and data characteristics.

        Parameters
        -----------
        df : Union[pd.Series, pd.DataFrame, dd.DataFrame]
            DataFrame for the operation

        Returns
        --------
        str
            Unique cache key
        """
        from pamola_core.utils.ops.op_cache import operation_cache

        # Get operation parameters
        parameters = self._get_base_parameters()

        # Fingerprint of everything that decides the result (see _cache_data_hash)
        data_hash = self._cache_data_hash(df)

        # Use the operation_cache utility to generate a consistent cache key
        return operation_cache.generate_cache_key(
            operation_name=self.operation_name,
            parameters=parameters,
            data_hash=data_hash,
        )


class FieldOperation(BaseOperation, ABC):
    """
    Base class for operations that process specific fields.

    Extends :class:`BaseOperation` to handle per-field logic while inheriting
    configuration, performance, and output settings from the base class.
    """

    def __init__(self, field_name: str, **kwargs):
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

        Parameters
        -----------
        field_name : str
            Name of the related field
        """
        self.scope.add_field(field_name)

    def validate_field_existence(self, df: DataFrameType) -> bool:
        """
        Validate that the main field exists in the DataFrame.

        Parameters
        -----------
        df : DataFrameType
            DataFrame to check

        Returns
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

        Parameters
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

        Parameters
        -----------
        group_name : str
            Name of the field group

        Returns
        --------
        List[str]
            List of field names in the group
        """
        return self.scope.get_fields_in_group(group_name)
