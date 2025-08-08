"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Validation Package Initialization
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Central initialization file for the validation package. Provides
  organized imports and exports for all validation components,
  making them easily accessible through a single import point.

Package Structure:
  - base.py: Core validation infrastructure (ValidationResult, BaseValidator)
  - decorators.py: Validation decorators for common patterns
  - exceptions.py: Custom exception classes for validation errors
  - field_validators.py: Validators for different field types
  - file_validators.py: File and path validation utilities
  - strategy_validators.py: Anonymization strategy validators
  - type_validators.py: Specialized type validators (network, geo, etc.)

Usage:
  The validation package can be used in two ways:

  1. Direct imports from submodules:
     from pamola_core.anonymization.commons.validation.field_validators import NumericFieldValidator
     from pamola_core.anonymization.commons.validation.exceptions import ValidationError

  2. Through the facade (validation_utils.py) for backward compatibility:
     from pamola_core.anonymization.commons import validation_utils
     result = validation_utils.validate_numeric_field(df, "age")

Design Principles:
  - Modular: Each module handles a specific validation domain
  - Consistent: All validators return ValidationResult objects
  - Extensible: Easy to add new validators and validation types
  - Compatible: Maintains backward compatibility through facade

Key Exports:
  This __init__.py exports all public classes, functions, and constants
  from the validation submodules, organized by category for easy discovery.
"""

# =============================================================================
# Core Infrastructure (from base.py)
# =============================================================================
from .base import (
    # Core result class
    ValidationResult,

    # Base validator classes
    BaseValidator,
    CompositeValidator,

    # Context and caching
    ValidationContext,
    ValidationCache,

    # Utility functions
    check_field_exists,
    check_multiple_fields_exist,
    is_numeric_type,
    is_categorical_type,
    is_datetime_type,
    safe_sample,
    validate_type,
    validate_range,

    # Decorators from base
    cached_validation,
    validation_handler as base_validation_handler  # Renamed to avoid conflict
)

# =============================================================================
# Validation Decorators (from decorators.py)
# =============================================================================
from .decorators import (
    # Core decorators
    validation_handler,
    validate_types,

    # Input handling
    sanitize_inputs,
    skip_if_empty,
    requires_field,
    requires_fields,

    # Composite decorators
    standard_validator,

    # Result handling
    aggregate_results
)

# =============================================================================
# Exceptions (from exceptions.py)
# =============================================================================
from .exceptions import (
    # Base exception
    ValidationError,

    # Field exceptions
    FieldNotFoundError,
    FieldTypeError,
    FieldValueError,

    # Strategy exceptions
    InvalidStrategyError,
    InvalidParameterError,

    # Conditional exceptions
    ConditionalValidationError,

    # Data type exceptions
    InvalidDataFormatError,
    RangeValidationError,

    # File exceptions
    FileValidationError,
    FileNotFoundError as FileNotFoundValidationError,  # Avoid conflict with builtin
    InvalidFileFormatError,

    # Aggregated exceptions
    ValidationErrorInfo,
    MultipleValidationErrors,

    # Configuration exceptions
    ConfigurationError,

    # Helper functions
    raise_if_errors
)

# =============================================================================
# Field Validators (from field_validators.py)
# =============================================================================
from .field_validators import (
    # Validator classes
    NumericFieldValidator,
    CategoricalFieldValidator,
    DateTimeFieldValidator,
    BooleanFieldValidator,
    TextFieldValidator,
    FieldExistsValidator,
    PatternValidator,
    # Factory function
    create_field_validator
)

# =============================================================================
# File Validators (from file_validators.py)
# =============================================================================
from .file_validators import (
    # Path validators
    FilePathValidator,
    DirectoryPathValidator,

    # Configuration file validators
    JSONFileValidator,
    CSVFileValidator,
    HierarchyFileValidator,

    # Multi-file validator
    MultiFileValidator,

    # Convenience functions
    validate_file_path,
    validate_directory_path
)

# =============================================================================
# Strategy Validators (from strategy_validators.py)
# =============================================================================
from .strategy_validators import (
    # Strategy validation functions
    validate_strategy,
    validate_generalization_strategy,
    validate_noise_strategy,
    validate_suppression_strategy,
    validate_masking_strategy,
    validate_pseudonymization_strategy,

    # Mode validation functions
    validate_operation_mode,
    validate_null_strategy,

    # Parameter validation functions
    validate_bin_count,
    validate_precision,
    validate_range_limits,
    validate_percentiles,

    # Strategy-specific validators
    validate_noise_parameters,
    validate_masking_parameters,
    validate_hierarchy_parameters,

    # Composite validators
    validate_strategy_compatibility,
    validate_output_field_configuration,

    # Strategy constants
    GENERALIZATION_STRATEGIES,
    NOISE_STRATEGIES,
    SUPPRESSION_STRATEGIES,
    MASKING_STRATEGIES,
    PSEUDONYMIZATION_STRATEGIES,
    OPERATION_MODES,
    NULL_STRATEGIES
)

# =============================================================================
# Type Validators (from type_validators.py)
# =============================================================================
from .type_validators import (
    # Network validators
    NetworkValidator,
    validate_network_identifiers,

    # Geographic validators
    GeographicValidator,
    validate_geographic_data,

    # Temporal validators
    TemporalValidator,
    validate_temporal_sequence,

    # Financial validators
    FinancialValidator,
    validate_financial_data,

    # Composite validator
    SpecializedTypeValidator,
    validate_specialized_type,

    # Pattern constants
    NETWORK_PATTERNS,
    GEO_PATTERNS,
    FINANCIAL_PATTERNS,
    COMMON_CURRENCIES
)


# =============================================================================
# Convenience Functions for Common Use Cases
# =============================================================================

def validate_field(df, field_name, field_type=None, **kwargs):
    """
    Convenience function to validate a field based on its type.

    Args:
        df: DataFrame containing the field
        field_name: Name of the field to validate
        field_type: Type of field (if None, will be inferred)
        **kwargs: Additional validation parameters

    Returns:
        ValidationResult object
    """
    # Check field existence first
    if not check_field_exists(df, field_name):
        return ValidationResult(
            is_valid=False,
            field_name=field_name,
            errors=[f"Field '{field_name}' not found in DataFrame"]
        )

    # Infer field type if not provided
    if field_type is None:
        series = df[field_name]
        if is_numeric_type(series):
            field_type = 'numeric'
        elif is_datetime_type(series):
            field_type = 'datetime'
        elif is_categorical_type(series):
            field_type = 'categorical'
        else:
            field_type = 'text'

    # Create and run appropriate validator
    validator = create_field_validator(field_type, **kwargs)
    return validator.validate(df[field_name], field_name=field_name)


def validate_operation_config(operation_type, strategy, mode, field_type, **params):
    """
    Convenience function to validate complete operation configuration.

    Args:
        operation_type: Type of operation (generalization, noise, etc.)
        strategy: Strategy name
        mode: Operation mode (REPLACE/ENRICH)
        field_type: Type of field being processed
        **params: Strategy-specific parameters

    Returns:
        ValidationResult aggregating all validation checks
    """
    result = ValidationResult(is_valid=True)

    # Validate mode
    mode_result = validate_operation_mode(mode)
    result.merge(mode_result)

    # Validate strategy compatibility
    compat_result = validate_strategy_compatibility(strategy, field_type, operation_type)
    result.merge(compat_result)

    # Validate strategy-specific parameters
    if operation_type == 'generalization' and strategy == 'binning':
        if 'bin_count' in params:
            bin_result = validate_bin_count(params['bin_count'])
            result.merge(bin_result)
    elif operation_type == 'generalization' and strategy == 'rounding':
        if 'precision' in params:
            prec_result = validate_precision(params['precision'])
            result.merge(prec_result)
    # Add more parameter validations as needed

    return result


# =============================================================================
# Module Metadata
# =============================================================================
__version__ = "1.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# =============================================================================
# Public API Definition
# =============================================================================
# This defines what gets imported with "from validation import *"
# We intentionally exclude some internal utilities to keep the API clean
__all__ = [
    # Core infrastructure
    'ValidationResult',
    'BaseValidator',
    'CompositeValidator',
    'ValidationContext',
    'ValidationCache',

    # Core decorators
    'validation_handler',
    'validate_types',
    'standard_validator',
    'cached_validation',

    # Core exceptions
    'ValidationError',
    'FieldNotFoundError',
    'FieldTypeError',
    'FieldValueError',
    'InvalidStrategyError',
    'InvalidParameterError',

    # Field validators
    'NumericFieldValidator',
    'CategoricalFieldValidator',
    'DateTimeFieldValidator',
    'BooleanFieldValidator',
    'TextFieldValidator',
    'FieldExistsValidator',
    'PatternValidator',
    # Factory function for field validators
    'create_field_validator',

    # File validators
    'FilePathValidator',
    'DirectoryPathValidator',
    'JSONFileValidator',
    'CSVFileValidator',
    'HierarchyFileValidator',
    'validate_file_path',
    'validate_directory_path',

    # Strategy validators
    'validate_strategy',
    'validate_operation_mode',
    'validate_null_strategy',
    'validate_bin_count',
    'validate_precision',
    'validate_strategy_compatibility',

    # Type validators
    'NetworkValidator',
    'GeographicValidator',
    'TemporalValidator',
    'FinancialValidator',
    'validate_specialized_type',

    # Convenience functions
    'validate_field',
    'validate_operation_config',

    # Utility functions
    'check_field_exists',
    'check_multiple_fields_exist',
    'is_numeric_type',
    'is_categorical_type',
    'is_datetime_type',

    # Constants
    'GENERALIZATION_STRATEGIES',
    'NOISE_STRATEGIES',
    'OPERATION_MODES',
    'NULL_STRATEGIES'
]

# =============================================================================
# Package Initialization
# =============================================================================
# Any package-level initialization can be done here
# For now, we just ensure the logger is configured
import logging

logger = logging.getLogger(__name__)
logger.debug("Validation package initialized successfully")