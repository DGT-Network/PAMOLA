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
