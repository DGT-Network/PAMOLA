"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Configuration
Description: Configuration management for operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides classes for managing operation configuration,
including parameter validation, storage, and serialization.

Key features:
- JSON Schema validation (REQ-OPS-002)
- Configuration serialization and deserialization
- Type-safe configuration access
- Registry for operation configuration classes

Satisfies:
REQ-OPS-002: Provides schema validation for operation parameters.
REQ-OPS-004: Supports serialization of configuration to JSON.

TODO:
- Fully migrate JSON schema validation to pamola_core.utils.io_helpers.json_utils.validate_json_schema()
- Consider standardizing error classes with OpsError base class
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Generic, Union

from pamola_core.common.enum.form_groups import GroupName

# Configure logger
logger = logging.getLogger(__name__)

# Type variable for configuration classes
T = TypeVar("T")


class OpsError(Exception):
    """Base class for all operation-related errors."""

    pass


class ConfigError(OpsError):
    """Error related to configuration operations."""

    pass


class OperationConfig(Generic[T]):
    """
    Base class for operation configuration.

    This class provides utilities for validating, storing, and serializing
    operation configuration parameters.

    Satisfies REQ-OPS-004: Provides configuration management for operations.
    """

    # JSON Schema for configuration validation
    schema: Dict[str, Any] = {"type": "object", "properties": {}}

    def __init__(self, **kwargs):
        """
        Initialize configuration with parameters.

        Parameters:
        -----------
        **kwargs : dict
            Configuration parameters
        """
        self._validate_params(kwargs)
        self._params = kwargs

    def _validate_params(self, params: Dict[str, Any]) -> None:
        """
        Validate parameters against the schema.

        Parameters:
        -----------
        params : Dict[str, Any]
            Parameters to validate

        Raises:
        -------
        ConfigError
            If parameters don't conform to schema.

        Satisfies:
        ----------
        REQ-OPS-002: Provides schema validation for operation parameters.
        """
        # Use the helper function from json_utils for validation
        from pamola_core.utils.io_helpers.json_utils import validate_json_schema

        validate_json_schema(params, self.schema, ConfigError)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to save the configuration file

        Satisfies:
        ----------
        REQ-OPS-004: Supports saving configuration to JSON.
        """
        path = Path(path) if isinstance(path, str) else path
        with open(path, "w") as f:
            json.dump(self._params, f, indent=2)  # type: ignore

    @classmethod
    def load(cls: Type[T], path: Union[str, Path]) -> T:
        """
        Load configuration from a JSON file.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the configuration file

        Returns:
        --------
        OperationConfig
            Loaded configuration

        Raises:
        -------
        ConfigError
            If the loaded data doesn't conform to schema.

        Satisfies:
        ----------
        REQ-OPS-004: Supports loading configuration from JSON.
        """
        path = Path(path) if isinstance(path, str) else path
        with open(path, "r") as f:
            params = json.load(f)
        return cls(**params)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration parameter.

        Parameters:
        -----------
        key : str
            Parameter name
        default : Any, optional
            Default value if parameter is not found

        Returns:
        --------
        Any
            Parameter value or default
        """
        return self._params.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a parameter by key."""
        return self._params[key]

    def __contains__(self, key: str) -> bool:
        """Check if a parameter exists."""
        return key in self._params

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
        --------
        Dict[str, Any]
            Dictionary representation of the configuration
        """
        return self._params.copy()

    def update(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to update

        Raises:
        -------
        ConfigError
            If the updated parameters don't conform to schema.
        """
        # Validate new parameters
        self._validate_params(kwargs)

        # Update parameters
        self._params.update(kwargs)

    def __repr__(self) -> str:
        """String representation of the configuration."""
        param_str = ", ".join(f"{k}={repr(v)}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({param_str})"


class OperationConfigRegistry:
    """
    Registry for operation configuration classes.

    This class allows registering and retrieving configuration classes
    for different operation types.
    """

    _registry: Dict[str, Type[OperationConfig]] = {}

    @classmethod
    def register(cls, operation_type: str, config_class: Type[OperationConfig]) -> None:
        """
        Register a configuration class for an operation type.

        Parameters:
        -----------
        operation_type : str
            Operation type identifier
        config_class : Type[OperationConfig]
            Configuration class to register
        """
        cls._registry[operation_type] = config_class
        logger.debug(
            f"Registered configuration class for operation type: {operation_type}"
        )

    @classmethod
    def get_config_class(cls, operation_type: str) -> Optional[Type[OperationConfig]]:
        """
        Get the configuration class for an operation type.

        Parameters:
        -----------
        operation_type : str
            Operation type identifier

        Returns:
        --------
        Type[OperationConfig] or None
            Configuration class, or None if not found
        """
        return cls._registry.get(operation_type)

    @classmethod
    def create_config(cls, operation_type: str, **kwargs) -> Optional[OperationConfig]:
        """
        Create a configuration instance for an operation type.

        Parameters:
        -----------
        operation_type : str
            Operation type identifier
        **kwargs : dict
            Configuration parameters

        Returns:
        --------
        OperationConfig or None
            Configuration instance, or None if operation type not found
        """
        config_class = cls.get_config_class(operation_type)
        if config_class is None:
            logger.warning(
                f"No configuration class registered for operation type: {operation_type}"
            )
            return None

        return config_class(**kwargs)


class BaseOperationConfig(OperationConfig):
    """
    Configuration schema for BaseOperation.

    Defines all shared operation parameters such as performance, output, and encryption.
    This base schema ensures consistency across all operation configurations.
    """

    schema = {
        "type": "object",
        "title": "Base Operation Configuration",
        "description": "Defines global parameters that control how operations are executed, optimized, and output.",
        "properties": {
            # --- Metadata ---
            "name": {
                "type": "string",
                "title": "Operation Name",
                "description": "Human-readable name of the operation.",
                "default": "",
            },
            "description": {
                "type": "string",
                "title": "Description",
                "description": "Optional detailed description of this operation.",
                "default": "",
            },
            "scope": {
                "type": ["object", "null"],
                "title": "Execution Scope",
                "description": "Optional scope or context within which the operation will execute.",
            },
            "config": {
                "type": ["object", "null"],
                "title": "Custom Configuration",
                "description": "Additional configuration parameters, typically used internally.",
            },
            # --- Performance & Processing ---
            "optimize_memory": {
                "type": "boolean",
                "x-component": "Checkbox",
                "title": "Optimize Memory Usage",
                "description": "If true, operations will use memory-efficient data structures.",
                "default": True,
            },
            "adaptive_chunk_size": {
                "type": "boolean",
                "title": "Adaptive Chunk Size",
                "x-component": "Checkbox",
                "description": "Automatically adjust chunk size based on data volume and system resources.",
                "default": True,
            },
            "mode": {
                "type": "string",
                "x-component": "Select",
                "oneOf": [
                    {"const": "REPLACE", "description": "Replace"},
                    {"const": "ENRICH", "description": "Enrich"},
                ],
                "title": "Mode",
                "description": "Defines how results will be applied to the dataset: REPLACE overwrites, ENRICH adds new data.",
                "default": "REPLACE",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "output_field_name": {
                "type": ["string", "null"],
                "title": "Output Field Name",
                "x-component": "Input",
                "description": "Optional custom name for the generated or modified output field.",
                "default": "",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                "x-depend-on": {"mode": "ENRICH"},
            },
            "column_prefix": {
                "type": "string",
                "title": "Column Prefix",
                "x-component": "Input",
                "description": "Prefix to apply to newly generated columns.",
                "default": "_",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                "x-depend-on": {"mode": "ENRICH", "output_field_name": "null"},
            },
            "null_strategy": {
                "type": "string",
                "oneOf": [
                    {"const": "PRESERVE", "description": "Preserve"},
                    {"const": "EXCLUDE", "description": "Exclude"},
                    {"const": "ANONYMIZE", "description": "Anonymize"},
                    {"const": "ERROR", "description": "Error"},
                ],
                "x-component": "Select",
                "title": "Handle Null Values",
                "description": "Determines how null or missing values are handled during processing.",
                "default": "PRESERVE",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "engine": {
                "type": "string",
                "oneOf": [
                    {"const": "auto", "description": "Auto"},
                    {"const": "pandas", "description": "Pandas"},
                    {"const": "dask", "description": "Dask"},
                ],
                "x-component": "Select",
                "title": "Execution Engine",
                "description": "Execution backend used to process data. 'auto' selects the best engine automatically.",
                "default": "auto",
            },
            "use_dask": {
                "type": "boolean",
                "title": "Enable Dask Processing",
                "x-component": "Checkbox",
                "description": "If true, operations are distributed across multiple Dask workers.",
                "default": False,
            },
            "npartitions": {
                "type": ["integer", "null"],
                "title": "Number of Dask Partitions",
                "x-component": "NumberPicker",
                "description": "Number of partitions to split the dataset into for parallel processing.",
                "minimum": 1,
            },
            "dask_partition_size": {
                "type": ["string", "null"],
                "title": "Dask Partition Size",
                "x-component": "Input",
                "description": "Approximate size of each Dask partition (e.g. '100MB').",
                "default": "100MB",
            },
            "use_vectorization": {
                "type": "boolean",
                "x-component": "Checkbox",
                "title": "Enable Vectorization",
                "description": "Use NumPy vectorized operations for faster computation where applicable.",
                "default": False,
            },
            "parallel_processes": {
                "type": ["integer", "null"],
                "title": "Parallel Processes",
                "x-component": "NumberPicker",
                "description": "Number of CPU processes to use for parallel execution.",
                "minimum": 1,
            },
            "chunk_size": {
                "type": "integer",
                "title": "Chunk Size",
                "x-component": "NumberPicker",
                "description": "Number of rows to process per batch when streaming or chunked processing is enabled.",
                "minimum": 1,
                "default": 10000,
            },
            # --- Output ---
            "use_cache": {
                "type": "boolean",
                "title": "Use Result Cache",
                "x-component": "Checkbox",
                "description": "Cache the operation output to speed up repeated runs with the same inputs.",
                "default": False,
            },
            "output_format": {
                "type": "string",
                "oneOf": [
                    {"const": "csv", "description": "csv"},
                    {"const": "parquet", "description": "parquet"},
                    {"const": "json", "description": "json"},
                ],
                "x-component": "Select",
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
                "title": "Output Format",
                "description": "Format used when saving processed output data.",
                "default": "csv",
            },
            "visualization_theme": {
                "type": ["string", "null"],
                "title": "Visualization Theme",
                "x-component": "Input",
                "description": "Optional color or layout theme for visualizations.",
            },
            "visualization_backend": {
                "type": ["string", "null"],
                "enum": ["plotly", "matplotlib", None],
                "oneOf": [
                    {"const": "plotly", "description": "Plotly"},
                    {"const": "matplotlib", "description": "Matplotlib"},
                ],
                "x-component": "Select",
                "title": "Visualization Backend",
                "description": "Rendering backend for generated plots and charts.",
                "default": "plotly",
            },
            "visualization_strict": {
                "type": "boolean",
                "title": "Strict Visualization Mode",
                "x-component": "Checkbox",
                "description": "If true, visualization errors will stop execution instead of being ignored.",
                "default": False,
            },
            "visualization_timeout": {
                "type": "integer",
                "title": "Visualization Timeout (seconds)",
                "x-component": "NumberPicker",
                "description": "Maximum time allowed for generating visualization before timing out.",
                "minimum": 1,
                "default": 120,
            },
            # --- Security & Encryption ---
            "use_encryption": {
                "type": "boolean",
                "title": "Enable Encryption",
                "x-component": "Checkbox",
                "description": "Encrypt sensitive data outputs using the selected encryption mode.",
                "default": False,
            },
            "encryption_mode": {
                "type": ["string", "null"],
                "oneOf": [
                    {"const": "age", "description": "Age"},
                    {"const": "simple", "description": "Simple"},
                    {"const": "none", "description": "None"},
                ],
                "x-component": "Select",
                "title": "Encryption Mode",
                "description": "Algorithm used for encrypting outputs. 'none' disables encryption.",
                "default": "none",
            },
            "encryption_key": {
                "type": ["string", "null"],
                "title": "Encryption Key",
                "x-component": "Input",
                "description": "Key or passphrase used for encryption when enabled.",
            },
            # --- Runtime & Execution Control ---
            "force_recalculation": {
                "type": "boolean",
                "title": "Force Recalculation",
                "x-component": "Checkbox",
                "description": "Re-run operation even if cached results exist.",
                "default": False,
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "generate_visualization": {
                "type": "boolean",
                "title": "Generate Visualization",
                "x-component": "Checkbox",
                "description": "If true, automatically generate visualization outputs after processing.",
                "default": True,
                "x-group": GroupName.OPERATION_BEHAVIOR_OUTPUT,
            },
            "save_output": {
                "type": "boolean",
                "title": "Save Output",
                "x-component": "Checkbox",
                "description": "If true, persist processed data to disk or database.",
                "default": True,
            },
        },
    }
