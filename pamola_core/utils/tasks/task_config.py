"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Task Configuration
Description: Configuration loading and management for tasks
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for loading and managing task configurations,
including cascading from project configuration, task-specific configuration,
and command-line overrides.

Key features:
- Project root and data repository discovery
- Configuration loading with priority cascade
- Path resolution for task directories
- Configuration validation
- Support for environment variables
- Secure handling of sensitive configuration
"""

import json
import logging
import os
import warnings
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import yaml

from pamola_core.utils.io import read_json, ensure_directory, write_json
from pamola_core.utils.tasks.path_security import validate_path_security, PathSecurityError
from pamola_core.utils.tasks.project_config_loader import (
    find_project_root, load_project_config
)
from pamola_core.utils.tasks.progress_manager import TaskProgressManager

# Set up logger
logger = logging.getLogger(__name__)

# Constants for progress tracking
STEP_FIND_PROJECT_ROOT = 1
STEP_LOAD_PROJECT_CONFIG = 2
STEP_LOAD_TASK_CONFIG = 3
STEP_CREATE_TASK_CONFIG = 4
STEP_VALIDATE_CONFIG = 5
TOTAL_CONFIG_STEPS = 5  # Total number of steps in configuration loading

# Default values
DEFAULT_CONFIG_DIR = "configs"
DEFAULT_DATA_REPOSITORY = "DATA"
DEFAULT_LOG_LEVEL = "INFO"
ENV_PREFIX = "PAMOLA_"


class EncryptionMode(Enum):
    """
    Encryption modes supported by the task framework.

    - NONE: No encryption
    - SIMPLE: Simple symmetric encryption
    - AGE: Age encryption (more secure, supports key rotation)
    """
    NONE = "none"
    SIMPLE = "simple"
    AGE = "age"

    @classmethod
    def from_string(cls, value: str) -> 'EncryptionMode':
        """Convert string to EncryptionMode enum value."""
        try:
            return cls(value.lower())
        except (ValueError, AttributeError):
            logger.warning(f"Invalid encryption mode '{value}', defaulting to 'simple'")
            return cls.SIMPLE


class ConfigurationError(Exception):
    """Exception raised for configuration-related errors."""
    pass


class DependencyMissingError(Exception):
    """Exception raised when required dependency is missing."""
    pass


class TaskConfig:
    """
    Task configuration container and manager.

    Holds configuration parameters for a task and provides
    methods for accessing them with the correct priority cascade:
    1. Command-line arguments
    2. Task-specific configuration
    3. Project-level configuration
    4. Built-in defaults
    5. Environment variables
    """

    def __init__(self,
                 config_dict: Dict[str, Any],
                 task_id: str,
                 task_type: str,
                 env_override: bool = True,
                 progress_manager: Optional[TaskProgressManager] = None):
        """
        Initialize configuration with values from dictionary.

        Args:
            config_dict: Dictionary containing configuration values
            task_id: ID of the task this configuration is for
            task_type: Type of the task this configuration is for
            env_override: Whether to allow environment variables to override config
            progress_manager: Optional progress manager for tracking initialization
        """
        self.task_id = task_id
        self.task_type = task_type
        self.progress_manager = progress_manager

        # Track original configuration for comparison
        self._original_config = config_dict.copy() if config_dict else {}

        # Set project root early for path resolution
        self.project_root = find_project_root()

        # Load project configuration
        if self.progress_manager:
            with self.progress_manager.create_operation_context(
                    name="load_config",
                    total=5,  # 5 steps in configuration loading
                    description="Loading task configuration"
            ) as progress:
                try:
                    # Step 1: Load project config
                    self.project_config = load_project_config(self.project_root)
                    progress.update(1, {"status": "project_config_loaded"})

                    # Step 2: Load base configuration
                    self._load_base_config(config_dict)
                    progress.update(1, {"status": "base_config_loaded"})

                    # Step 3: Set up directories
                    self._setup_directories()
                    progress.update(1, {"status": "directories_setup"})

                    # Step 4: Apply environment variable overrides if enabled
                    if env_override:
                        self._apply_env_overrides()
                    progress.update(1, {"status": "env_vars_applied"})

                    # Step 5: Set up additional properties and caches
                    self._setup_additional_properties()
                    progress.update(1, {"status": "setup_completed"})
                except Exception as e:
                    progress.update(1, {"status": "error", "error_message": str(e)})
                    raise
        else:
            try:
                # Load project configuration
                self.project_config = load_project_config(self.project_root)

                # Load base configuration
                self._load_base_config(config_dict)

                # Set up directories
                self._setup_directories()

                # Apply environment variable overrides if enabled
                if env_override:
                    self._apply_env_overrides()

                # Set up additional properties and caches
                self._setup_additional_properties()
            except FileNotFoundError:
                logger.warning(f"Project configuration file not found, using defaults")
                self.project_config = {}

    def _setup_additional_properties(self):
        """Set up additional properties and caches for the configuration."""
        # Track sensitive keys for secure handling
        self._sensitive_keys = {"encryption_key_path", "master_key_path"}

        # Cache for resolved paths
        self._path_cache: Dict[str, Path] = {}

        # Settings for external path access
        self.allow_external = self._original_config.get('allow_external', False)
        self.allowed_external_paths = self._original_config.get('allowed_external_paths', [])

        # Transition settings for backwards compatibility
        self.legacy_path_support = self._original_config.get('legacy_path_support', True)

        # Log configuration initialization
        logger.debug(f"Initialized TaskConfig for task {self.task_id} ({self.task_type})")

    def _load_base_config(self, config_dict: Dict[str, Any]) -> None:
        """
        Load base configuration from the config dictionary.

        Args:
            config_dict: Dictionary containing configuration values
        """
        # Extract base configuration with defaults from project config
        project_defaults = self.project_config.get("task_defaults", {})

        # Extract directory structure from project config
        dir_structure = self.project_config.get("directory_structure", {})
        data_repo = self.project_config.get("data_repository", DEFAULT_DATA_REPOSITORY)

        # Get task directory suffixes
        self.task_dir_suffixes = self.project_config.get("task_dir_suffixes", [
            "input", "output", "temp", "logs", "dictionaries"
        ])

        # Set data repository from cascade: config_dict -> project_config -> default
        self.data_repository = config_dict.get('data_repository', data_repo)

        # Set log level from cascade: config_dict -> project_config -> default
        self.log_level = config_dict.get(
            'log_level',
            self.project_config.get("logging", {}).get("level", DEFAULT_LOG_LEVEL)
        )

        # Extract directory structure with defaults from project config
        self.raw_dir = dir_structure.get('raw', 'raw')
        self.processed_dir = dir_structure.get('processed', 'processed')
        self.logs_dir = dir_structure.get('logs', 'logs')
        self.reports_dir = dir_structure.get('reports', 'reports')

        # Extract task-specific configuration from task section in project config
        project_task_config = self.project_config.get('tasks', {}).get(self.task_id, {})

        # Merge with task-specific configuration from config_dict
        task_config = {**project_task_config, **config_dict.get('tasks', {}).get(self.task_id, {})}

        # Apply cascade: task_config -> project_defaults
        self.dependencies = task_config.get('dependencies', project_defaults.get('dependencies', []))
        self.continue_on_error = task_config.get('continue_on_error', project_defaults.get('continue_on_error', False))

        # Extract encryption configuration from cascade
        project_encryption = self.project_config.get('encryption', {})
        self.use_encryption = task_config.get('use_encryption', project_encryption.get('use_encryption', False))
        self.encryption_key_path = task_config.get('encryption_key_path') or project_encryption.get('key_path')

        # Set encryption mode using the enum
        encryption_mode_str = (
                task_config.get('encryption_mode') or
                project_encryption.get('mode', 'simple' if self.use_encryption else 'none')
        )
        self.encryption_mode = EncryptionMode.from_string(encryption_mode_str)

        # Extract performance configuration with cascade
        project_performance = self.project_config.get('performance', {})
        self.use_vectorization = task_config.get(
            'use_vectorization',
            project_defaults.get('use_vectorization', False)
        )
        self.parallel_processes = task_config.get(
            'parallel_processes',
            project_defaults.get('parallel_processes', 1)
        )
        self.chunk_size = task_config.get(
            'chunk_size',
            project_performance.get('chunk_size', 100000)
        )
        self.default_encoding = task_config.get(
            'default_encoding',
            project_performance.get('default_encoding', 'utf-8')
        )
        self.default_delimiter = task_config.get(
            'default_delimiter',
            project_performance.get('default_delimiter', ',')
        )
        self.default_quotechar = task_config.get(
            'default_quotechar',
            project_performance.get('default_quotechar', '"')
        )
        self.memory_limit_mb = task_config.get(
            'memory_limit_mb',
            project_performance.get('memory_limit_mb', 1000)
        )
        self.use_dask = task_config.get(
            'use_dask',
            project_performance.get('use_dask', False)
        )

        # Extract scope configuration
        self.scope = task_config.get('scope', {})
        if 'fields' in task_config:
            self.scope['fields'] = task_config['fields']
        if 'datasets' in task_config:
            self.scope['datasets'] = task_config['datasets']
        if 'field_groups' in task_config:
            self.scope['field_groups'] = task_config['field_groups']

        # Add all additional task-specific configuration as attributes
        for key, value in task_config.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def _setup_directories(self) -> None:
        """
        Set up directory paths based on configuration.
        """
        # Data repository can be absolute or relative to project root
        data_repo_path = Path(self.data_repository)
        if not data_repo_path.is_absolute():
            data_repo_path = self.project_root / data_repo_path
        self.data_repository_path = data_repo_path

        # Set up pamola core directories
        self.raw_data_path = self._resolve_path(self.data_repository_path, self.raw_dir)
        self.processed_data_path = self._resolve_path(self.data_repository_path, self.processed_dir)
        self.reports_path = self._resolve_path(self.data_repository_path, self.reports_dir)

        # Set up task directory
        self.task_dir = self._resolve_path(self.processed_data_path, self.task_id)

        # Set up standard task subdirectories based on suffixes
        for suffix in self.task_dir_suffixes:
            dir_path = self._resolve_path(self.task_dir, suffix)
            setattr(self, f"{suffix}_dir", dir_path)
            # Note: directories are not created here to avoid I/O during initialization
            # They will be created on-demand when needed

        # For backward compatibility, explicitly set output_directory
        self.output_directory = self.task_dir

        # Set up report path
        self.report_path = self._resolve_path(
            self.reports_path,
            f"{self.task_id}_report.json"
        )

        # Set up log directory according to spec: always in project_root/logs
        self.log_directory = self._resolve_path(
            self.project_root,
            self.logs_dir
        )

        # Set up log file path
        self.log_file = self.log_directory / f"{self.task_id}.log"

        # Set up task-specific log directory
        self.task_log_directory = self._resolve_path(self.task_dir, "logs")

        # Set up task-specific log file
        self.task_log_file = self.task_log_directory / f"{self.task_id}.log"

    def _resolve_path(self, *parts) -> Path:
        """
        Resolve a path from components.

        Args:
            *parts: Path components to join

        Returns:
            Path: Resolved path
        """
        # Convert all parts to strings first
        str_parts = [str(part) for part in parts]

        # Use Path to create the path object
        path = Path(*str_parts)

        # Ensure path is absolute
        if not path.is_absolute():
            path = self.project_root / path

        return path

    def resolve_legacy_path(self, path: Union[str, Path]) -> Path:
        """
        Resolve a path using legacy format during transition period.
        Issues deprecation warning.

        Args:
            path: Path in legacy format

        Returns:
            Resolved absolute path
        """
        path_obj = Path(path)

        # Don't apply special handling to absolute paths
        if path_obj.is_absolute():
            if not validate_path_security(
                    path_obj,
                    allowed_paths=self.allowed_external_paths,
                    allow_external=self.allow_external
            ):
                raise PathSecurityError(f"Insecure absolute path: {path_obj}")
            return path_obj

        # Issue warning for legacy relative path usage
        warnings.warn(
            f"Using legacy relative path '{path}'. "
            f"Please use the path API methods instead (get_task_dir(), etc.).",
            DeprecationWarning, stacklevel=2
        )

        # Resolve relative to project root
        return self.project_root / path_obj

    def _apply_env_overrides(self) -> None:
        """
        Apply overrides from environment variables.

        Environment variables take the form PAMOLA_TASK_{TASK_ID}_{KEY}
        or PAMOLA_{KEY} for global settings.
        """
        # Get all environment variables with the PAMOLA prefix
        env_vars = {k: v for k, v in os.environ.items()
                    if k.startswith(ENV_PREFIX)}

        # Process task-specific variables first (higher priority)
        task_prefix = f"{ENV_PREFIX}TASK_{self.task_id.upper()}_"
        task_vars = {k[len(task_prefix):].lower(): v for k, v in env_vars.items()
                     if k.startswith(task_prefix)}

        # Process global variables (lower priority)
        global_vars = {k[len(ENV_PREFIX):].lower(): v for k, v in env_vars.items()
                       if k.startswith(ENV_PREFIX) and not k.startswith(task_prefix)}

        # Apply global variables first, then task-specific ones (for override priority)
        self._apply_env_dict(global_vars)
        self._apply_env_dict(task_vars)

    def _apply_env_dict(self, env_dict: Dict[str, str]) -> None:
        """
        Apply environment variable dictionary to configuration.

        Args:
            env_dict: Dictionary of environment variables
        """
        for key, value in env_dict.items():
            # Skip if key already exists from higher priority source
            if key in self._original_config:
                continue

            # Convert value to appropriate type
            converted_value = self._convert_env_value(value)

            # Set attribute
            logger.debug(f"Setting config {key}={converted_value} from environment")
            setattr(self, key, converted_value)

    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate Python type.

        Args:
            value: String value from environment variable

        Returns:
            Converted value with appropriate type
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        if value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Handle None values
        if value.lower() in ('none', 'null'):
            return None

        # Handle integer values
        try:
            if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                return int(value)
        except ValueError:
            pass

        # Handle float values
        try:
            return float(value)
        except ValueError:
            pass

        # Handle list values (comma-separated)
        if ',' in value:
            return [self._convert_env_value(v.strip()) for v in value.split(',')]

        # Default to string
        return value

    def override_with_args(self, args: Dict[str, Any]) -> None:
        """
        Override configuration with command line arguments.

        Args:
            args: Command line arguments
        """
        if not args:
            return

        needs_cache_clear = False

        # Override data repository if specified
        if 'data_repository' in args and args['data_repository']:
            self.data_repository = args['data_repository']
            needs_cache_clear = True

            # Update paths that depend on data repository
            self._setup_directories()

        # Override log level if specified
        if 'log_level' in args and args['log_level']:
            self.log_level = args['log_level']

        # Override encryption settings if specified
        if 'use_encryption' in args:
            self.use_encryption = args['use_encryption']

        if 'encryption_key_path' in args:
            self.encryption_key_path = args['encryption_key_path']

        if 'encryption_mode' in args:
            self.encryption_mode = EncryptionMode.from_string(args['encryption_mode'])

        # Override performance settings if specified
        if 'use_vectorization' in args:
            self.use_vectorization = args['use_vectorization']

        if 'parallel_processes' in args:
            self.parallel_processes = args['parallel_processes']

        if 'use_dask' in args:
            self.use_dask = args['use_dask']

        if 'chunk_size' in args:
            self.chunk_size = args['chunk_size']

        # Override scope fields if specified
        if 'fields' in args:
            if not hasattr(self, 'scope'):
                self.scope = {}
            self.scope['fields'] = args['fields']

        # Override scope datasets if specified
        if 'datasets' in args:
            if not hasattr(self, 'scope'):
                self.scope = {}
            self.scope['datasets'] = args['datasets']

        # Override continue_on_error if specified
        if 'continue_on_error' in args:
            self.continue_on_error = args['continue_on_error']

        # Override security settings if specified
        if 'allow_external' in args:
            self.allow_external = args['allow_external']
            needs_cache_clear = True

        if 'allowed_external_paths' in args:
            self.allowed_external_paths = args['allowed_external_paths']
            needs_cache_clear = True

        # Process any additional arguments that match attributes
        for key, value in args.items():
            if key not in [
                'data_repository', 'log_level', 'use_encryption',
                'encryption_key_path', 'encryption_mode', 'use_vectorization',
                'parallel_processes', 'fields', 'datasets',
                'continue_on_error', 'use_dask', 'chunk_size',
                'allow_external', 'allowed_external_paths'
            ] and value is not None:
                setattr(self, key, value)

        # Clear path cache after overrides if needed
        if needs_cache_clear:
            self._path_cache = {}
            logger.debug("Path cache cleared due to configuration changes")

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate the configuration.

        Returns:
            Tuple containing:
                - Boolean indicating whether configuration is valid
                - List of validation error messages
        """
        errors = []

        # Check required fields
        required_fields = ["task_id", "task_type"]
        for field in required_fields:
            if not hasattr(self, field) or getattr(self, field) is None:
                errors.append(f"Missing required field: {field}")

        # Validate directories
        if hasattr(self, 'data_repository_path'):
            if not self.data_repository_path.exists():
                try:
                    # Try to create the data repository if it doesn't exist
                    ensure_directory(self.data_repository_path)
                    logger.info(f"Created data repository directory: {self.data_repository_path}")
                except Exception as e:
                    errors.append(
                        f"Could not create data repository path: {self.data_repository_path}, error: {str(e)}")

        # Validate encryption settings consistency
        if hasattr(self, 'use_encryption') and self.use_encryption:
            if not hasattr(self, 'encryption_key_path') or not self.encryption_key_path:
                errors.append("Encryption is enabled (use_encryption=True) but no encryption_key_path is provided")

            if hasattr(self, 'encryption_mode') and self.encryption_mode == EncryptionMode.NONE:
                errors.append("Encryption is enabled (use_encryption=True) but encryption_mode is set to 'none'")

        # Validate task directory suffixes
        if not isinstance(self.task_dir_suffixes, list) or not all(isinstance(s, str) for s in self.task_dir_suffixes):
            errors.append("task_dir_suffixes must be a list of strings")

        # Return validation result
        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Dictionary representation of configuration
        """
        result = {}

        # Get all attributes that don't start with underscore
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)

                # Convert Path objects to strings
                if isinstance(value, Path):
                    value = str(value)
                # Convert Enum to string
                elif isinstance(value, Enum):
                    value = value.value

                result[key] = value

        return result

    def save(self, path: Optional[Path] = None, format: str = "json") -> Path:
        """
        Save configuration to file.

        Args:
            path: Path to save configuration file, or None to use default
            format: Format to save in - "json" or "yaml"

        Returns:
            Path to saved configuration file
        """
        if path is None:
            # Use default path in configs directory
            suffix = ".json" if format.lower() == "json" else ".yaml"
            path = self._resolve_path(
                self.project_root,
                DEFAULT_CONFIG_DIR,
                f"{self.task_id}{suffix}"
            )

        # Ensure directory exists
        ensure_directory(path.parent)

        # Convert to dictionary for serialization
        config_dict = self.to_dict()

        # Remove paths and runtime-specific settings
        keys_to_remove = ['project_root', 'data_repository_path', 'output_directory',
                          'report_path', 'log_directory', 'log_file', 'project_config',
                          'raw_data_path', 'processed_data_path', 'reports_path',
                          'task_log_directory', 'task_log_file', 'progress_manager']
        for key in keys_to_remove:
            if key in config_dict:
                del config_dict[key]

        # Redact sensitive information
        for key in self._sensitive_keys:
            if key in config_dict and config_dict[key] is not None:
                config_dict[key] = f"<redacted: {key[:3]}...>"

        # Write to file
        try:
            if format.lower() == "yaml":
                # PyYAML is already imported at the top of the module; check availability
                if yaml is not None:
                    with open(path, "w", encoding="utf-8") as f:
                        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
                else:
                    logger.warning("PyYAML not available - falling back to JSON format.")
                    write_json(config_dict, path)
            else:
                write_json(config_dict, path)

            logger.info(f"Configuration saved to {path}")
            return path
        except Exception as e:
            logger.error(f"Error saving configuration to {path}: {str(e)}")
            raise ConfigurationError(f"Failed to save configuration: {str(e)}")

    # Path API methods

    def get_project_root(self) -> Path:
        """Get the project root directory."""
        return self.project_root

    def get_data_repository(self) -> Path:
        """Get the data repository path."""
        return self.data_repository_path

    def get_raw_dir(self) -> Path:
        """Get the raw data directory."""
        return self.raw_data_path

    def get_processed_dir(self) -> Path:
        """Get the processed data directory."""
        return self.processed_data_path

    def get_reports_dir(self) -> Path:
        """Get the reports directory."""
        return self.reports_path

    def get_task_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task directory
        """
        if task_id is None:
            return self.task_dir

        # Cache key for resolving paths
        cache_key = f"task_dir_{task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = self._resolve_path(self.processed_data_path, task_id)

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def get_task_input_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task input directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task input directory
        """
        task_dir = self.get_task_dir(task_id)

        # Cache key for resolving paths
        cache_key = f"task_input_dir_{task_id or self.task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = task_dir / "input"

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def get_task_output_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task output directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task output directory
        """
        task_dir = self.get_task_dir(task_id)

        # Cache key for resolving paths
        cache_key = f"task_output_dir_{task_id or self.task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = task_dir / "output"

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def get_task_temp_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task temporary directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task temporary directory
        """
        task_dir = self.get_task_dir(task_id)

        # Cache key for resolving paths
        cache_key = f"task_temp_dir_{task_id or self.task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = task_dir / "temp"

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def get_task_dict_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task dictionaries directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task dictionaries directory
        """
        task_dir = self.get_task_dir(task_id)

        # Cache key for resolving paths
        cache_key = f"task_dict_dir_{task_id or self.task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = task_dir / "dictionaries"

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def get_task_logs_dir(self, task_id: Optional[str] = None) -> Path:
        """
        Get the task logs directory.

        Args:
            task_id: Optional task ID. If None, uses this task's ID.

        Returns:
            Path to the task logs directory
        """
        task_dir = self.get_task_dir(task_id)

        # Cache key for resolving paths
        cache_key = f"task_logs_dir_{task_id or self.task_id}"
        if cache_key in self._path_cache:
            return self._path_cache[cache_key]

        # Resolve path
        path = task_dir / "logs"

        # Cache result
        self._path_cache[cache_key] = path
        return path

    def processed_subdir(self, task_id: Optional[str] = None, *parts) -> Path:
        """
        Get a subdirectory within the processed directory for a specific task.

        Args:
            task_id: Task identifier (uses current task if None)
            *parts: Additional path components

        Returns:
            Path to the specified subdirectory
        """
        task_dir = self.get_task_dir(task_id)
        return task_dir.joinpath(*parts)

    def get_dependency_output(self, dependency_id: str, file_pattern: Optional[str] = None) -> Union[Path, List[Path]]:
        """
        Get the output directory or files from a dependency.

        Args:
            dependency_id: Dependency ID (task ID) or absolute path
            file_pattern: Optional file pattern to match within the dependency output dir

        Returns:
            Path to the dependency output directory or list of matching files

        Raises:
            PathSecurityError: If the path fails security validation
            DependencyMissingError: If the dependency output directory doesn't exist
        """
        # Check if dependency_id contains a path separator, treat as absolute path
        if os.path.sep in dependency_id or ':' in dependency_id:
            path = Path(dependency_id)

            # Validate path security with allowed paths
            if not validate_path_security(
                    path,
                    allowed_paths=self.allowed_external_paths,
                    allow_external=self.allow_external
            ):
                raise PathSecurityError(f"Absolute dependency path failed security validation: {path}")

            if not path.exists():
                error_message = f"Dependency path doesn't exist: {path}"
                if self.continue_on_error:
                    logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                    if file_pattern:
                        return []  # Return empty list for files
                    return path  # Return path even if it doesn't exist
                else:
                    raise DependencyMissingError(error_message)

            return path

        # Treat as task ID
        output_dir = self.get_task_output_dir(dependency_id)

        if not output_dir.exists():
            error_message = f"Dependency output directory doesn't exist: {output_dir}"
            if self.continue_on_error:
                logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                if file_pattern:
                    return []  # Return empty list for files
                return output_dir  # Return directory even if it doesn't exist
            else:
                raise DependencyMissingError(error_message)

        # If file pattern is specified, return matching files
        if file_pattern:
            matching_files = list(output_dir.glob(file_pattern))
            if not matching_files:
                logger.warning(f"No files matching pattern '{file_pattern}' in {output_dir}")
            return matching_files

        return output_dir

    def get_dependency_report(self, dependency_id: str) -> Path:
        """
        Get the report file for a dependency.

        Args:
            dependency_id: Dependency ID (task ID)

        Returns:
            Path to the dependency report file

        Raises:
            DependencyMissingError: If the dependency report doesn't exist
        """
        report_path = self._resolve_path(self.reports_path, f"{dependency_id}_report.json")

        if not report_path.exists():
            error_message = f"Dependency report doesn't exist: {report_path}"
            if self.continue_on_error:
                logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                return report_path  # Return path even if it doesn't exist
            else:
                raise DependencyMissingError(error_message)

        return report_path

    def assert_dependencies_completed(self) -> bool:
        """
        Check if all dependencies have completed successfully.

        Returns:
            True if all dependencies are complete, False otherwise

        Raises:
            DependencyMissingError: If a dependency report is missing or indicates failure
        """
        for dependency_id in self.dependencies:
            try:
                report_path = self.get_dependency_report(dependency_id)

                # Load report and check status
                try:
                    report_data = read_json(report_path)
                    if not report_data.get("success", False):
                        error_message = f"Dependency {dependency_id} failed: {report_data.get('error_info', {}).get('message', 'Unknown error')}"
                        if self.continue_on_error:
                            logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                        else:
                            raise DependencyMissingError(error_message)
                except json.JSONDecodeError:
                    error_message = f"Invalid report format for dependency {dependency_id}"
                    if self.continue_on_error:
                        logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                    else:
                        raise DependencyMissingError(error_message)

            except FileNotFoundError:
                error_message = f"Missing report for dependency {dependency_id}"
                if self.continue_on_error:
                    logger.warning(f"{error_message} (continuing due to continue_on_error=True)")
                else:
                    raise DependencyMissingError(error_message)

        return True

    # Scope methods

    def get_scope_fields(self) -> List[str]:
        """
        Get fields defined in the scope.

        Returns:
            List of field names in the scope
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('fields', [])
        return []

    def get_scope_datasets(self) -> List[str]:
        """
        Get datasets defined in the scope.

        Returns:
            List of dataset names in the scope
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('datasets', [])
        return []

    def get_scope_field_groups(self) -> Dict[str, List[str]]:
        """
        Get field groups defined in the scope.

        Returns:
            Dictionary mapping group names to lists of field names
        """
        if hasattr(self, 'scope') and isinstance(self.scope, dict):
            return self.scope.get('field_groups', {})
        return {}

    def __str__(self) -> str:
        """String representation of the configuration."""
        # Format key attributes but omit paths and sensitive information
        keys_to_show = ['task_id', 'task_type', 'data_repository', 'log_level',
                        'continue_on_error', 'use_encryption', 'encryption_mode',
                        'use_vectorization', 'parallel_processes']

        attributes = []
        for key in keys_to_show:
            if hasattr(self, key):
                value = getattr(self, key)
                if isinstance(value, Enum):
                    value = value.value
                attributes.append(f"{key}={value}")

        return f"TaskConfig({', '.join(attributes)})"


def load_task_config(
        task_id: str,
        task_type: str,
        args: Optional[Dict[str, Any]] = None,
        default_config: Optional[Dict[str, Any]] = None,
        progress_manager: Optional[Any] = None
) -> 'TaskConfig':
    """
    Load task configuration from project configuration file and override
    with command line arguments.

    The configuration loading follows this priority cascade:
    1. Command-line arguments (highest priority)
    2. Task-specific JSON configuration (if exists, completely overrides project config)
    3. If no JSON exists:
       a. Default configuration from task class
       b. Project-level configuration overrides in tasks.{task_id} section
       c. Save combined config to task-specific JSON for future runs

    Args:
        task_id: ID of the task
        task_type: Type of the task
        args: Command line arguments to override configuration
        default_config: Default configuration values from task class
        progress_manager: Optional progress manager for tracking configuration loading

    Returns:
        TaskConfig instance with loaded configuration

    Raises:
        ConfigurationError: If configuration cannot be loaded or validated
    """
    if progress_manager:
        with progress_manager.create_operation_context(
                name="load_task_config",
                total=TOTAL_CONFIG_STEPS,
                description="Loading task configuration",
                unit="steps"
        ) as progress:
            try:
                # Step 1: Find project root
                project_root = find_project_root()
                progress.update(1, {"status": "project_root_found"})

                # Step 2: Load project config
                try:
                    project_config = load_project_config(project_root)

                    # Ensure enable_checkpoints exists in task_defaults with proper default
                    task_defaults = project_config.get("task_defaults", {})
                    if "enable_checkpoints" not in task_defaults:
                        task_defaults["enable_checkpoints"] = False
                        project_config["task_defaults"] = task_defaults

                    progress.update(1, {"status": "project_config_loaded"})
                except Exception as e:
                    logger.warning(f"Failed to load project configuration: {e}")
                    # Create minimal project config with default enable_checkpoints setting
                    project_config = {"task_defaults": {"enable_checkpoints": False}}
                    progress.update(1, {"status": "project_config_failed"})

                # Extract task-specific overrides from project config
                project_task_config = project_config.get('tasks', {}).get(task_id, {})

                # Step 3: Try to load task-specific config JSON
                task_config_path = project_root / DEFAULT_CONFIG_DIR / f"{task_id}.json"
                should_bootstrap = True

                if task_config_path.exists():
                    # JSON exists - attempt to load and use it exclusively for this task
                    try:
                        task_specific_config = read_json(task_config_path)

                        # Ensure enable_checkpoints has a default value if not specified in JSON config
                        task_specific_config.setdefault(
                            "enable_checkpoints",
                            project_config.get("task_defaults", {}).get("enable_checkpoints", False)
                        )

                        logger.debug(f"Loaded task configuration from {task_config_path}")
                        should_bootstrap = False
                        progress.update(1, {"status": "task_config_loaded"})
                    except Exception as e:
                        logger.warning(f"Failed to load task configuration from {task_config_path}: {e}")
                        should_bootstrap = True
                        progress.update(1, {"status": "task_config_failed"})

                # If JSON loading failed or file doesn't exist, bootstrap configuration
                if should_bootstrap:
                    logger.debug(f"Task configuration bootstrap needed for {task_id}")

                    # Start with default config if provided (using deep copy)
                    if default_config is not None:
                        task_specific_config = deepcopy(default_config)
                        logger.debug("Using default configuration from task class")
                    else:
                        task_specific_config = {}
                        logger.debug("No default configuration available")

                    # Apply project-level overrides from tasks.{task_id} section
                    if project_task_config:
                        task_specific_config.update(project_task_config)
                        logger.debug("Applied project-level overrides from YAML")

                    # Ensure enable_checkpoints has a default value if not specified in task config
                    task_specific_config.setdefault(
                        "enable_checkpoints",
                        project_config.get("task_defaults", {}).get("enable_checkpoints", False)
                    )

                    # Save the bootstrapped config to JSON for future runs
                    try:
                        ensure_directory(task_config_path.parent)
                        write_json(task_specific_config, task_config_path)
                        logger.info(f"Created new task configuration file at {task_config_path}")
                    except Exception as e:
                        logger.warning(f"Failed to create task configuration file at {task_config_path}: {e}")

                    # Ensure we properly update progress
                    if not task_config_path.exists():
                        progress.update(1, {"status": "bootstrap_config_created"})

                # Build the final configuration dictionary
                config_dict = {}

                # First add project-level configuration
                project_defaults = project_config.get("task_defaults", {})
                config_dict["task_defaults"] = project_defaults

                # Ensure tasks dictionary exists
                if "tasks" not in config_dict:
                    config_dict["tasks"] = {}

                # Add task-specific configuration
                config_dict["tasks"][task_id] = task_specific_config

                # Add top-level references for direct access
                # This replaces any keys from project config, ensuring task config has highest priority
                for key, value in task_specific_config.items():
                    config_dict[key] = value

                # Step 4: Create task configuration
                task_config = TaskConfig(config_dict, task_id, task_type, progress_manager=progress_manager)
                progress.update(1, {"status": "task_config_created"})

                # Override with command line arguments
                if args:
                    task_config.override_with_args(args)
                    logger.debug("Configuration overridden with command line arguments")

                # Step 5: Validate configuration
                is_valid, errors = task_config.validate()
                if not is_valid:
                    logger.warning(f"Configuration validation failed: {'; '.join(errors)}")
                    progress.update(1, {"status": "validation_warning", "errors": errors})
                else:
                    progress.update(1, {"status": "validation_success"})

                return task_config

            except Exception as e:
                # Update progress with error status
                progress.update(1, {"status": "error", "error_message": str(e)})
                logger.error(f"Error loading task configuration: {e}", exc_info=True)
                raise ConfigurationError(f"Failed to load task configuration: {e}")
    else:
        # Non-progress-tracked path
        try:
            # Find project root
            project_root = find_project_root()

            # Task-specific config path
            task_config_path = project_root / DEFAULT_CONFIG_DIR / f"{task_id}.json"

            # Load project config
            try:
                project_config = load_project_config(project_root)

                # Ensure enable_checkpoints exists in task_defaults with proper default
                task_defaults = project_config.get("task_defaults", {})
                if "enable_checkpoints" not in task_defaults:
                    task_defaults["enable_checkpoints"] = False
                    project_config["task_defaults"] = task_defaults

                logger.debug(f"Loaded project configuration")
            except Exception as e:
                logger.warning(f"Failed to load project configuration: {e}")
                # Create minimal project config with default enable_checkpoints setting
                project_config = {"task_defaults": {"enable_checkpoints": False}}

            # Extract task-specific overrides from project config
            project_task_config = project_config.get('tasks', {}).get(task_id, {})
            should_bootstrap = True

            # Try to load task-specific config JSON
            if task_config_path.exists():
                # JSON exists - attempt to load and use it exclusively for this task
                try:
                    task_specific_config = read_json(task_config_path)

                    # Ensure enable_checkpoints has a default value if not specified in JSON config
                    task_specific_config.setdefault(
                        "enable_checkpoints",
                        project_config.get("task_defaults", {}).get("enable_checkpoints", False)
                    )

                    logger.debug(f"Loaded task configuration from {task_config_path}")
                    should_bootstrap = False
                except Exception as e:
                    logger.warning(f"Failed to load task configuration from {task_config_path}: {e}")
                    should_bootstrap = True

            # If JSON loading failed or file doesn't exist, bootstrap configuration
            if should_bootstrap:
                logger.debug(f"Task configuration bootstrap needed for {task_id}")

                # Start with default config if provided (using deep copy)
                if default_config is not None:
                    task_specific_config = deepcopy(default_config)
                    logger.debug("Using default configuration from task class")
                else:
                    task_specific_config = {}
                    logger.debug("No default configuration available")

                # Apply project-level overrides from tasks.{task_id} section
                if project_task_config:
                    task_specific_config.update(project_task_config)
                    logger.debug("Applied project-level overrides from YAML")

                # Ensure enable_checkpoints has a default value if not specified in task config
                task_specific_config.setdefault(
                    "enable_checkpoints",
                    project_config.get("task_defaults", {}).get("enable_checkpoints", False)
                )

                # Save the bootstrapped config to JSON for future runs
                try:
                    ensure_directory(task_config_path.parent)
                    write_json(task_specific_config, task_config_path)
                    logger.info(f"Created new task configuration file at {task_config_path}")
                except Exception as e:
                    logger.warning(f"Failed to create task configuration file at {task_config_path}: {e}")

            # Build the final configuration dictionary
            config_dict = {}

            # First add project-level configuration
            project_defaults = project_config.get("task_defaults", {})
            config_dict["task_defaults"] = project_defaults

            # Ensure tasks dictionary exists
            if "tasks" not in config_dict:
                config_dict["tasks"] = {}

            # Add task-specific configuration
            config_dict["tasks"][task_id] = task_specific_config

            # Add top-level references for direct access
            # This replaces any keys from project config, ensuring task config has highest priority
            for key, value in task_specific_config.items():
                config_dict[key] = value

            # Create task configuration
            task_config = TaskConfig(config_dict, task_id, task_type)

            # Override with command line arguments
            if args:
                task_config.override_with_args(args)
                logger.debug("Configuration overridden with command line arguments")

            # Validate configuration
            is_valid, errors = task_config.validate()
            if not is_valid:
                logger.warning(f"Configuration validation failed: {'; '.join(errors)}")

            return task_config

        except Exception as e:
            logger.error(f"Error loading task configuration: {e}", exc_info=True)
            raise ConfigurationError(f"Failed to load task configuration: {e}")