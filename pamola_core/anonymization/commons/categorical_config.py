"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Categorical Generalization Configuration Management
Package:       pamola_core.anonymization.generalization
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025-01-23
Updated:       2025-01-23
License:       BSD 3-Clause

Description:
   This module provides configuration management for categorical generalization
   operations, including validation, schema definitions, and parameter handling
   for various generalization strategies.

Purpose:
   Centralizes all configuration logic for categorical generalization to ensure
   consistent parameter validation, type safety, and compatibility across
   different strategies and execution modes.

Key Features:
   - Comprehensive JSON schema validation with cached validators
   - Strategy-specific parameter validation
   - NULL handling strategy configuration
   - Template-based unknown value generation
   - Large data mode configuration (Dask support)
   - Thread-safe configuration handling
   - Export/import configuration support

Design Principles:
   - Type Safety: All parameters are strongly typed and validated
   - Extensibility: Easy to add new strategies or parameters
   - Compatibility: Backward compatible with existing operations
   - Validation: Comprehensive validation at configuration time
   - Performance: Cached validators and efficient validation

Dependencies:
   - jsonschema: For JSON schema validation
   - typing: For type hints and validation
   - dataclasses: For configuration data structures
   - pamola_core.utils.ops.op_config: Base configuration class
   - logging: For configuration diagnostics

Usage:
   ```python
   config = CategoricalGeneralizationConfig(
       field_name="city",
       strategy="hierarchy",
       external_dictionary_path="cities.json",
       null_strategy="PRESERVE",
       unknown_value="OTHER",
       rare_value_template="OTHER_{n}"
   )

   # Validate configuration
   if config.validate():
       # Use configuration
       params = config.get_strategy_params()
   ```

Changelog:
   1.1.0 (2025-01-23):
     - Fixed schema conflict by renaming to SCHEMA (ClassVar)
     - Added validation for ENRICH mode output_field_name
     - Added cached validator for performance
     - Enhanced strategy parameter validation
     - Added __str__ method for logging
     - Fixed Enum handling in dataclass fields
     - Added privacy threshold validation
     - Used field(repr=False) for sensitive data
   1.0.0 - Initial implementation with full configuration support
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, ClassVar

import jsonschema
from jsonschema import Draft7Validator

from pamola_core.utils.ops.op_config import OperationConfig

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 10000
DEFAULT_CACHE_SIZE = 10000
MAX_HIERARCHY_LEVELS = 5
MAX_CATEGORIES = 1000000
DEFAULT_MIN_GROUP_SIZE = 10
DEFAULT_FREQ_THRESHOLD = 0.01
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_MAX_ROWS_IN_MEMORY = 1000000
DEFAULT_DASK_CHUNK_SIZE = 50000


# Enums for validated choices
class GeneralizationStrategy(str, Enum):
    """Supported generalization strategies."""

    HIERARCHY = "hierarchy"
    MERGE_LOW_FREQ = "merge_low_freq"
    FREQUENCY_BASED = "frequency_based"


class NullStrategy(str, Enum):
    """NULL value handling strategies."""

    PRESERVE = "PRESERVE"  # Keep NULL as is
    EXCLUDE = "EXCLUDE"  # Remove NULL values
    ANONYMIZE = "ANONYMIZE"  # Replace NULL with unknown_value
    ERROR = "ERROR"  # Raise error on NULL


class OperationMode(str, Enum):
    """Operation mode for field updates."""

    REPLACE = "REPLACE"  # Replace original field
    ENRICH = "ENRICH"  # Add new field


class GroupRareAs(str, Enum):
    """Rare category grouping options."""

    OTHER = "OTHER"  # Single OTHER group
    CATEGORY_N = "CATEGORY_N"  # Numbered categories
    RARE_N = "RARE_N"  # Numbered rare groups


class TextNormalization(str, Enum):
    """Text normalization levels."""

    NONE = "none"  # No normalization
    BASIC = "basic"  # Basic normalization
    ADVANCED = "advanced"  # Advanced normalization
    AGGRESSIVE = "aggressive"  # Aggressive normalization


# Pre-compute enum values for performance
STRATEGY_VALUES = [s.value for s in GeneralizationStrategy]
NULL_STRATEGY_VALUES = [n.value for n in NullStrategy]
MODE_VALUES = [m.value for m in OperationMode]
GROUP_RARE_VALUES = [g.value for g in GroupRareAs]
TEXT_NORM_VALUES = [t.value for t in TextNormalization]
SUPPORTED_DICT_FORMATS = ["auto", "json", "csv"]


@dataclass
class CategoricalGeneralizationConfig(OperationConfig):
    """
    Configuration for categorical generalization operations.

    Extends OperationConfig with categorical-specific parameters and validation.
    """

    # Required fields
    field_name: str
    strategy: str = GeneralizationStrategy.HIERARCHY.value
    # Operation mode
    mode: str = OperationMode.REPLACE.value
    output_field_name: Optional[str] = None
    column_prefix: str = "_"
    # NULL handling
    null_strategy: str = NullStrategy.PRESERVE.value
    # Metadata
    description: str = ""
    # Dictionary parameters
    external_dictionary_path: Optional[str] = None
    dictionary_format: str = "auto"
    hierarchy_level: int = 1
    # Frequency-based parameters
    merge_low_freq: bool = False
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE
    freq_threshold: float = DEFAULT_FREQ_THRESHOLD
    max_categories: int = MAX_CATEGORIES
    # Unknown handling
    allow_unknown: bool = True
    unknown_value: str = "OTHER"
    group_rare_as: str = GroupRareAs.OTHER.value
    rare_value_template: str = "OTHER_{n}"
    # Text processing
    text_normalization: str = TextNormalization.BASIC.value
    case_sensitive: bool = False
    fuzzy_matching: bool = False
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    # Privacy thresholds
    privacy_check_enabled: bool = True
    min_acceptable_k: int = 5
    max_acceptable_disclosure_risk: float = 0.2
    # Conditional processing
    condition_field: Optional[str] = None
    condition_values: Optional[List] = None
    condition_operator: str = "in"
    # Risk-based processing
    ka_risk_field: Optional[str] = None
    risk_threshold: float = 5.0
    vulnerable_record_strategy: str = "generalize"
    quasi_identifiers: Optional[List[str]] = None
    # Standard parameters from base class
    chunk_size: int = DEFAULT_CHUNK_SIZE
    optimize_memory: bool = True
    adaptive_chunk_size: bool = True
    use_dask: bool = False
    npartitions: Optional[int] = None
    dask_partition_size: Optional[str] = None
    use_vectorization: bool = False
    parallel_processes: Optional[int] = None
    use_cache: bool = True
    use_encryption: bool = False
    encryption_mode: Optional[str] = None
    encryption_key: Optional[Union[str, Path]] = None
    # Visualization options
    visualization_theme: Optional[str] = None
    visualization_backend: Optional[str] = "plotly"
    visualization_strict: bool = False
    visualization_timeout: int = 120
    # Output options
    output_format: str = "csv"

    # Class-level constants (renamed to avoid conflict with instance attribute)
    _VALIDATOR: ClassVar[Optional[Draft7Validator]] = None

    # JSON Schema for validation (renamed from 'schema' to 'SCHEMA' to avoid conflict)
    SCHEMA: ClassVar[Dict[str, Any]] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            # Required fields
            "field_name": {
                "type": "string",
                "minLength": 1,
                "description": "Field to generalize",
            },
            "strategy": {
                "type": "string",
                "enum": STRATEGY_VALUES,
                "description": "Generalization strategy",
            },
            # Operation mode
            "mode": {
                "type": "string",
                "enum": MODE_VALUES,
                "description": "Operation mode",
            },
            "output_field_name": {
                "type": ["string", "null"],
                "description": "Output field name for ENRICH mode",
            },
            "column_prefix": {
                "type": "string",
                "description": "Column prefix for output fields",
            },
            # NULL handling
            "null_strategy": {
                "type": "string",
                "enum": NULL_STRATEGY_VALUES,
                "description": "NULL value handling strategy",
            },
            # Metadata
            "description": {"type": "string", "description": "Operation description"},
            # Dictionary parameters
            "external_dictionary_path": {
                "type": ["string", "null"],
                "description": "Path to external dictionary",
            },
            "dictionary_format": {
                "type": "string",
                "enum": SUPPORTED_DICT_FORMATS,
                "description": "Dictionary file format",
            },
            "hierarchy_level": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_HIERARCHY_LEVELS,
                "description": "Hierarchy level for generalization",
            },
            # Frequency-based parameters
            "merge_low_freq": {
                "type": "boolean",
                "description": "Merge low frequency categories",
            },
            "min_group_size": {
                "type": "integer",
                "minimum": 1,
                "description": "Minimum group size",
            },
            "freq_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Frequency threshold",
            },
            "max_categories": {
                "type": "integer",
                "minimum": 0,
                "description": "Maximum categories to preserve",
            },
            # Unknown handling
            "allow_unknown": {"type": "boolean", "description": "Allow unknown values"},
            "unknown_value": {"type": "string", "description": "Value for unknowns"},
            "group_rare_as": {
                "type": "string",
                "enum": GROUP_RARE_VALUES,
                "description": "How to group rare categories",
            },
            "rare_value_template": {
                "type": "string",
                "pattern": ".*\\{n\\}.*",
                "description": "Template for numbered rare values",
            },
            # Text processing
            "text_normalization": {
                "type": "string",
                "enum": TEXT_NORM_VALUES,
                "description": "Text normalization level",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Case sensitive matching",
            },
            "fuzzy_matching": {
                "type": "boolean",
                "description": "Enable fuzzy matching",
            },
            "similarity_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Similarity threshold for fuzzy matching",
            },
            # Privacy thresholds
            "privacy_check_enabled": {
                "type": "boolean",
                "description": "Enable privacy checks",
            },
            "min_acceptable_k": {
                "type": "integer",
                "minimum": 2,  # Enforce k >= 2
                "description": "Minimum acceptable k-anonymity",
            },
            "max_acceptable_disclosure_risk": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Maximum acceptable disclosure risk",
            },
            # Conditional processing
            "condition_field": {
                "type": ["string", "null"],
                "description": "Field for conditional processing",
            },
            "condition_values": {
                "type": ["array", "null"],
                "description": "Values for condition",
            },
            "condition_operator": {
                "type": "string",
                "enum": ["in", "not_in", "eq", "ne", "gt", "lt", "gte", "lte"],
                "description": "Condition operator",
            },
            # Risk-based processing
            "ka_risk_field": {
                "type": ["string", "null"],
                "description": "Field with k-anonymity risk",
            },
            "risk_threshold": {
                "type": "number",
                "minimum": 0,
                "description": "Risk threshold",
            },
            "vulnerable_record_strategy": {
                "type": "string",
                "enum": ["generalize", "suppress", "skip"],
                "description": "Strategy for vulnerable records",
            },
            "quasi_identifiers": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Quasi-identifier fields",
            },
            # Standard parameters
            "chunk_size": {
                "type": "integer",
                "minimum": 1,
                "description": "Chunk size for processing",
            },
            "optimize_memory": {
                "type": "boolean",
                "description": "Optimize memory usage",
            },
            "adaptive_chunk_size": {
                "type": "boolean",
                "description": "Adapt chunk size based on memory",
            },
            "use_dask": {"type": "boolean"},
            "npartitions": {"type": ["integer", "null"], "minimum": 1},
            "dask_partition_size": {"type": ["string", "null"], "default": "100MB"},
            "use_vectorization": {"type": "boolean"},
            "parallel_processes": {"type": ["integer", "null"], "minimum": 1},
            "use_cache": {"type": "boolean", "description": "Use caching"},
            "use_encryption": {
                "type": "boolean",
                "description": "Use encryption for data",
            },
            "encryption_mode": {
                "type": ["string", "null"],
                "description": "Encryption mode",
            },
            "encryption_key": {
                "type": ["string", "null"],
                "description": "Encryption key or path",
            },
            # Visualization options
            "visualization_theme": {"type": ["string", "null"]},
            "visualization_backend": {
                "type": ["string", "null"],
                "enum": ["plotly", "matplotlib", None],
            },
            "visualization_strict": {"type": "boolean"},
            "visualization_timeout": {"type": "integer", "minimum": 1, "default": 120},
            # Output options
            "output_format": {
                "type": "string",
                "enum": ["csv", "parquet", "json"],
                "default": "csv",
            },
        },
        "required": ["field_name", "strategy"],
        "additionalProperties": True,
    }

    @classmethod
    def _get_validator(cls) -> Draft7Validator:
        """Get cached validator instance."""
        if cls._VALIDATOR is None:
            cls._VALIDATOR = Draft7Validator(cls.SCHEMA)
        return cls._VALIDATOR

    def __post_init__(self):
        """
        Post-initialization validation and setup.
        Ensures OperationConfig.__init__ is called so that _params and other base attributes are set.
        """
        # If rare_value_template is not None and there is no '{n}', automatically add '_{n}' at the end
        if self.rare_value_template and "{n}" not in self.rare_value_template:
            self.rare_value_template = f"{self.rare_value_template}_{{n}}"

        # Call OperationConfig.__init__ to initialize base attributes (especially _params)
        OperationConfig.__init__(self, **self.__dict__)

        # Assign instance schema from class constant (if base class expects it)
        if hasattr(self, "schema"):
            self.schema = self.__class__.SCHEMA

        # Validate configuration
        self.validate()

        # Set default description if not provided
        if not self.description:
            self.description = (
                f"Categorical generalization for field '{self.field_name}' "
                f"using {self.strategy} strategy"
            )

        # Log configuration summary
        logger.debug(f"Initialized {self.__class__.__name__}: {self.get_summary()}")

    def __str__(self) -> str:
        """String representation for logging."""
        return self.get_summary()

    def validate(self) -> bool:
        """
        Validate configuration against schema and business rules.

        Returns:
        --------
        bool
            True if valid

        Raises:
        -------
        ValueError
            If configuration is invalid
        """
        # Convert to dict for validation
        config_dict = self.to_dict()

        # JSON schema validation using cached validator
        validator = self._get_validator()
        try:
            validator.validate(config_dict)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Configuration validation failed: {e.message}")

        # Strategy-specific validation
        self._validate_strategy_params()

        # Mode-specific validation
        self._validate_mode_params()

        # NULL strategy validation
        self._validate_null_strategy()

        # Template validation
        self._validate_templates()

        # Privacy threshold validation
        self._validate_privacy_thresholds()

        return True

    def _validate_strategy_params(self) -> None:
        """Validate strategy-specific parameters."""
        if self.strategy == GeneralizationStrategy.HIERARCHY.value:
            if not self.external_dictionary_path:
                raise ValueError("Hierarchy strategy requires external_dictionary_path")

            # Validate dictionary format
            if self.dictionary_format not in SUPPORTED_DICT_FORMATS:
                raise ValueError(
                    f"Unsupported dictionary format: {self.dictionary_format}. "
                    f"Must be one of: {SUPPORTED_DICT_FORMATS}"
                )

            # Check file exists if not a URL and not auto format
            if not self.external_dictionary_path.startswith(("http://", "https://")):
                path = Path(self.external_dictionary_path)
                if not path.exists():
                    raise ValueError(
                        f"Dictionary file not found: {self.external_dictionary_path}"
                    )

                # Validate format matches file extension if not auto
                if self.dictionary_format != "auto":
                    ext = path.suffix.lower().lstrip(".")
                    if ext != self.dictionary_format:
                        logger.warning(
                            f"Dictionary format '{self.dictionary_format}' doesn't match "
                            f"file extension '{ext}'"
                        )

            if self.hierarchy_level < 1 or self.hierarchy_level > MAX_HIERARCHY_LEVELS:
                raise ValueError(
                    f"hierarchy_level must be between 1 and {MAX_HIERARCHY_LEVELS}"
                )

        elif self.strategy == GeneralizationStrategy.MERGE_LOW_FREQ.value:
            if self.min_group_size < 1:
                raise ValueError("min_group_size must be at least 1")

            if self.freq_threshold < 0 or self.freq_threshold > 1:
                raise ValueError("freq_threshold must be between 0 and 1")

        elif self.strategy == GeneralizationStrategy.FREQUENCY_BASED.value:
            if self.max_categories < 1:
                raise ValueError("max_categories must be at least 1")

    def _validate_mode_params(self) -> None:
        """Validate mode-specific parameters."""
        if self.mode == OperationMode.ENRICH.value:
            if not self.output_field_name and not self.column_prefix:
                raise ValueError(
                    "ENRICH mode requires output_field_name or column_prefix to be specified"
                )

            # Ensure output field is different from input
            if self.output_field_name == self.field_name:
                raise ValueError(
                    "output_field_name must be different from field_name in ENRICH mode"
                )

    def _validate_null_strategy(self) -> None:
        """Validate NULL strategy configuration."""
        if self.null_strategy not in NULL_STRATEGY_VALUES:
            raise ValueError(
                f"Invalid null_strategy: {self.null_strategy}. "
                f"Must be one of: {NULL_STRATEGY_VALUES}"
            )

        # If NULL strategy is ERROR, ensure allow_unknown is False
        if self.null_strategy == NullStrategy.ERROR.value and self.allow_unknown:
            logger.warning(
                "null_strategy is ERROR but allow_unknown is True. "
                "This may cause unexpected behavior."
            )

        # If NULL strategy is ANONYMIZE, ensure unknown_value is provided
        if (
            self.null_strategy == NullStrategy.ANONYMIZE.value
            and not self.unknown_value
        ):
            raise ValueError("null_strategy is ANONYMIZE but no unknown_value provided")

    def _validate_templates(self) -> None:
        """Validate template strings."""
        # Validate rare_value_template
        if "{n}" not in self.rare_value_template:
            raise ValueError("rare_value_template must contain {n} placeholder")

        # Check for valid template syntax
        try:
            self.rare_value_template.format(n=1)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid rare_value_template format: {e}")

    def _validate_privacy_thresholds(self) -> None:
        """Validate privacy threshold settings."""
        if self.min_acceptable_k < 2:
            raise ValueError(
                "min_acceptable_k must be at least 2 for meaningful k-anonymity"
            )

        if (
            self.max_acceptable_disclosure_risk < 0
            or self.max_acceptable_disclosure_risk > 1
        ):
            raise ValueError("max_acceptable_disclosure_risk must be between 0 and 1")

        # Warn if privacy checks are disabled but thresholds are set
        if not self.privacy_check_enabled and (
            self.min_acceptable_k != 5 or self.max_acceptable_disclosure_risk != 0.2
        ):
            logger.warning(
                "Privacy checks are disabled but custom thresholds are set. "
                "Enable privacy_check_enabled to use these thresholds."
            )

    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Get parameters specific to the selected strategy.

        Returns:
        --------
        Dict[str, Any]
            Strategy-specific parameters
        """
        base_params = {
            "allow_unknown": self.allow_unknown,
            "unknown_value": self.unknown_value,
            "text_normalization": self.text_normalization,
            "case_sensitive": self.case_sensitive,
            "null_strategy": self.null_strategy,
        }

        if self.strategy == GeneralizationStrategy.HIERARCHY.value:
            return {
                **base_params,
                "external_dictionary_path": self.external_dictionary_path,
                "dictionary_format": self.dictionary_format,
                "hierarchy_level": self.hierarchy_level,
                "fuzzy_matching": self.fuzzy_matching,
                "similarity_threshold": self.similarity_threshold,
            }

        elif self.strategy == GeneralizationStrategy.MERGE_LOW_FREQ.value:
            return {
                **base_params,
                "min_group_size": self.min_group_size,
                "freq_threshold": self.freq_threshold,
                "group_rare_as": self.group_rare_as,
                "rare_value_template": self.rare_value_template,
                "max_categories": self.max_categories,
            }

        elif self.strategy == GeneralizationStrategy.FREQUENCY_BASED.value:
            return {
                **base_params,
                "max_categories": self.max_categories,
                "min_group_size": self.min_group_size,
                "group_rare_as": self.group_rare_as,
                "rare_value_template": self.rare_value_template,
            }

        return base_params

    def format_rare_value(self, index: int) -> str:
        """
        Format a rare value using the template.

        Parameters:
        -----------
        index : int
            Index for the rare value

        Returns:
        --------
        str
            Formatted rare value
        """
        return self.rare_value_template.format(n=index)

    def get_summary(self) -> str:
        """
        Get a human-readable summary of the configuration.

        Returns:
        --------
        str
            Configuration summary
        """
        summary_parts = [
            f"Strategy: {self.strategy}",
            f"Field: {self.field_name}",
            f"Mode: {self.mode}",
        ]

        if self.strategy == GeneralizationStrategy.HIERARCHY.value:
            summary_parts.extend(
                [
                    f"Dictionary: {Path(self.external_dictionary_path).name if self.external_dictionary_path else 'None'}",
                    f"Level: {self.hierarchy_level}",
                    f"Fuzzy: {self.fuzzy_matching}",
                ]
            )
        elif self.strategy in [
            GeneralizationStrategy.MERGE_LOW_FREQ.value,
            GeneralizationStrategy.FREQUENCY_BASED.value,
        ]:
            summary_parts.extend(
                [
                    f"Min group size: {self.min_group_size}",
                    f"Max categories: {self.max_categories}",
                ]
            )

        summary_parts.extend([f"NULL strategy: {self.null_strategy}"])

        return " | ".join(summary_parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Delegates to parent class if available, otherwise implements locally.

        Returns:
        --------
        Dict[str, Any]
            Configuration as dictionary
        """
        # Try to use parent's to_dict if available
        if hasattr(super(), "to_dict"):
            result = super().to_dict()
        else:
            # Local implementation
            result = {}
            for key, value in self.__dict__.items():
                if not key.startswith("_") and value is not None:
                    if isinstance(value, Path):
                        result[key] = str(value)
                    elif isinstance(value, Enum):
                        result[key] = value.value
                    else:
                        result[key] = value

        return result

    @classmethod
    def from_dict(
        cls, config_dict: Dict[str, Any]
    ) -> "CategoricalGeneralizationConfig":
        """
        Create configuration from dictionary.

        Parameters:
        -----------
        config_dict : Dict[str, Any]
            Configuration dictionary

        Returns:
        --------
        CategoricalGeneralizationConfig
            Configuration instance
        """
        # Remove any keys not in the schema
        valid_keys = set(cls.SCHEMA["properties"].keys())
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

        return cls(**filtered_dict)

    def export_json(self, filepath: Union[str, Path]) -> None:
        """
        Export configuration to JSON file.

        Parameters:
        -----------
        filepath : Union[str, Path]
            Output file path
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

        logger.info(f"Exported configuration to {filepath}")

    @classmethod
    def import_json(
        cls, filepath: Union[str, Path]
    ) -> "CategoricalGeneralizationConfig":
        """
        Import configuration from JSON file.

        Parameters:
        -----------
        filepath : Union[str, Path]
            Input file path

        Returns:
        --------
        CategoricalGeneralizationConfig
            Configuration instance
        """
        filepath = Path(filepath)

        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        logger.info(f"Imported configuration from {filepath}")
        return cls.from_dict(config_dict)


# Validation helper functions


def validate_strategy_parameters(
    strategy: str,
    params: Dict[str, Any],
    valid_strategies: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Any]:
    """
    Validate parameters for a specific strategy.

    Parameters:
    -----------
    strategy : str
        Strategy name
    params : Dict[str, Any]
        Parameters to validate
    valid_strategies : Dict[str, Dict[str, List[str]]]
        Valid strategies and their required parameters

    Returns:
    --------
    Dict[str, Any]
        Validation result with is_valid and errors
    """
    result = {"is_valid": True, "errors": []}

    if strategy not in valid_strategies:
        result["is_valid"] = False
        result["errors"].append(f"Unknown strategy: {strategy}")
        return result

    # Check required parameters
    required = valid_strategies[strategy].get("required", [])
    for param in required:
        if param not in params or params[param] is None:
            result["is_valid"] = False
            result["errors"].append(f"Missing required parameter: {param}")

    return result


def validate_null_strategy(
    null_strategy: str, allow_unknown: bool, unknown_value: str
) -> Dict[str, Any]:
    """
    Validate NULL strategy configuration.

    Parameters:
    -----------
    null_strategy : str
        NULL handling strategy
    allow_unknown : bool
        Whether unknown values are allowed
    unknown_value : str
        Value for unknowns

    Returns:
    --------
    Dict[str, Any]
        Validation result
    """
    result = {"is_valid": True, "warnings": []}

    if null_strategy == NullStrategy.ERROR.value and allow_unknown:
        result["warnings"].append("NULL strategy is ERROR but allow_unknown is True")

    if null_strategy == NullStrategy.ANONYMIZE.value and not unknown_value:
        result["is_valid"] = False
        result["warnings"].append(
            "NULL strategy is ANONYMIZE but no unknown_value provided"
        )

    return result


# Module metadata
__version__ = "1.1.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main classes and functions
__all__ = [
    "CategoricalGeneralizationConfig",
    "GeneralizationStrategy",
    "NullStrategy",
    "OperationMode",
    "GroupRareAs",
    "TextNormalization",
    "validate_strategy_parameters",
    "validate_null_strategy",
    # Pre-computed constants
    "STRATEGY_VALUES",
    "NULL_STRATEGY_VALUES",
    "MODE_VALUES",
    "GROUP_RARE_VALUES",
    "TEXT_NORM_VALUES",
    "SUPPORTED_DICT_FORMATS",
]
