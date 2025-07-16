"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Anonymization Validation Utilities (Complete Facade)
Package:       pamola_core.anonymization.commons
Version:       3.2.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
Updated:       2025-06-15
License:       BSD 3-Clause

Description:
   This module serves as a complete facade for the validation framework, providing
   backward compatibility while offering access to the new modular validation
   system. It re-exports all validation functions, decorators, and utilities
   for comprehensive validation support.

   The actual validation logic has been refactored into the validation
   subpackage for better maintainability and modularity.

Key Features:
   - Complete backward compatible API for existing code
   - Access to all validation functionality from modular system
   - Factory functions for creating validators
   - All decorators and utilities exported
   - Strategy validation with all constants
   - Unified error handling and result structures
   - Type checking utilities and helpers

Framework:
   This facade integrates with the modular validation framework located at
   pamola_core.anonymization.commons.validation/, providing a simplified interface
   while maintaining all functionality.

Migration Guide:
   Old: from pamola_core.anonymization.commons.validation_utils import validate_numeric_field
   New: from pamola_core.anonymization.commons.validation import NumericFieldValidator

   Both approaches work, but the new approach offers more flexibility.

Changelog:
   3.2.0 - 2025-06-15 - Complete facade with all missing functionality
         - Added all strategy validators and constants
         - Added all decorators from decorators module
         - Added base utilities and type checking functions
         - Fixed missing exports in __all__
         - Comprehensive re-export of all validation features
   3.1.0 - Fixed import issues and naming conflicts
         - Added pathlib import for Path type
         - Improved error handling for FileNotFoundError
         - Enhanced documentation and type hints
   3.0.0 - Complete refactoring into modular validation framework
         - Created facade for backward compatibility
         - Added factory functions and convenience methods
   2.1.0 - Enhanced categorical validation and hierarchy support
   2.0.0 - Added conditional processing and specialized validators
   1.0.0 - Initial monolithic implementation

Dependencies:
   - pandas - DataFrame operations
   - numpy - Numeric operations
   - logging - Error reporting
   - typing - Type hints
   - pathlib - Path validation
   - pamola_core.anonymization.commons.validation - New modular validation system
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Import from new modular validation system
# Note: Using explicit imports to avoid naming conflicts

# =============================================================================
# Base Infrastructure Imports
# =============================================================================
from .validation import (
    # Base infrastructure
    ValidationResult,
    BaseValidator,
    CompositeValidator,
    ValidationContext,
    ValidationCache,
    # Core Exceptions
    ValidationError,
    FieldNotFoundError,
    FieldTypeError,
    FieldValueError,
    InvalidStrategyError,
    InvalidParameterError,
    ConditionalValidationError,
    InvalidDataFormatError,
    RangeValidationError,
    FileValidationError,
    InvalidFileFormatError,
    ValidationErrorInfo,
    MultipleValidationErrors,
    ConfigurationError,
    raise_if_errors,
    # Base utilities
    is_numeric_type,
    is_categorical_type,
    is_datetime_type,
    safe_sample,
    validate_type,
    validate_range,
    cached_validation,
)

# =============================================================================
# Field Validator Imports
# =============================================================================
from .validation import (
    # Field validators
    NumericFieldValidator,
    CategoricalFieldValidator,
    DateTimeFieldValidator,
    BooleanFieldValidator,
    TextFieldValidator,
    create_field_validator,
)

# =============================================================================
# File Validator Imports
# =============================================================================
from .validation import (
    # File validators
    FilePathValidator,
    DirectoryPathValidator,
    HierarchyFileValidator,
    JSONFileValidator,
    CSVFileValidator,
    MultiFileValidator,
)

# =============================================================================
# Strategy Validator Imports
# =============================================================================
from .validation import (
    # Strategy validators (import with aliases to avoid conflicts)
    validate_strategy as validate_strategy_new,
    validate_generalization_strategy as validate_generalization_strategy_new,
    validate_noise_strategy,
    validate_suppression_strategy,
    validate_masking_strategy,
    validate_pseudonymization_strategy,
    # Mode validators
    validate_operation_mode,
    validate_null_strategy,
    # Parameter validators
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
    NULL_STRATEGIES,
)

# =============================================================================
# Type Validator Imports
# =============================================================================
from .validation import (
    # Type validators
    NetworkValidator,
    GeographicValidator,
    TemporalValidator,
    FinancialValidator,
    SpecializedTypeValidator,
    validate_specialized_type,
    # Pattern constants
    NETWORK_PATTERNS,
    GEO_PATTERNS,
    FINANCIAL_PATTERNS,
    COMMON_CURRENCIES,
)

# =============================================================================
# Decorator Imports
# =============================================================================
from .validation import (
    # Decorators
    validation_handler,
    standard_validator,
    validate_types,
    sanitize_inputs,
    skip_if_empty,
    requires_field,
    requires_fields,
    aggregate_results,
)

# =============================================================================
# Utility Function Imports
# =============================================================================
from .validation import (
    # Utility functions
    check_field_exists,
    check_multiple_fields_exist,
)

# Handle FileNotFoundError import - use custom validation error
try:
    from .validation.exceptions import FileNotFoundError as FileNotFoundValidationError
except ImportError:
    # Fallback if the exception doesn't exist in the new system
    FileNotFoundValidationError = FileNotFoundError

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Legacy Support Layer
# =============================================================================
# This section provides backward compatibility for old API calls


class LegacyValidationSupport:
    """
    Provides backward compatibility for legacy validation calls.

    This class wraps old-style validation functions to work with the new
    validation system while maintaining the original API.
    """

    @staticmethod
    def wrap_validation_result(
        result: ValidationResult, return_bool: bool = True
    ) -> Union[bool, Tuple[bool, Dict[str, Any]]]:
        """
        Convert new ValidationResult to legacy format.

        Args:
            result: ValidationResult from new system
            return_bool: If True, return only boolean; if False, return tuple

        Returns:
            Boolean or tuple based on legacy API expectations
        """
        if return_bool:
            return result.is_valid
        else:
            details = {
                "valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                **result.details,
            }
            return result.is_valid, details


# =============================================================================
# Backward Compatibility Functions
# =============================================================================


def validate_field_exists(
    df: pd.DataFrame, field_name: str, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Validate that a field exists in the DataFrame.

    DEPRECATED: Use check_field_exists() or requires_field decorator instead.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    field_name : str
        The name of the field to verify
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored, uses module logger)

    Returns:
    --------
    bool
        True if the field exists, False otherwise
    """
    log = logger_instance or logger
    if not check_field_exists(df, field_name):
        log.error(f"Field '{field_name}' does not exist in the DataFrame")
        return False
    return True


def validate_multiple_fields_exist(
    df: pd.DataFrame,
    field_names: List[str],
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate that multiple fields exist in the DataFrame.

    DEPRECATED: Use check_multiple_fields_exist() instead.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to check
    field_names : List[str]
        List of field names to verify
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored, uses module logger)

    Returns:
    --------
    Tuple[bool, List[str]]
        (True if all fields exist, list of missing fields)
    """
    return check_multiple_fields_exist(df, field_names)


@validation_handler()
def validate_numeric_field(
    df: pd.DataFrame,
    field_name: str,
    allow_null: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate that a field is numeric with optional range checking.

    DEPRECATED: Use NumericFieldValidator instead for more features.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    min_value : Optional[float]
        Minimum allowed value (default: None)
    max_value : Optional[float]
        Maximum allowed value (default: None)
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored, uses module logger)

    Returns:
    --------
    bool
        True if the field is numeric and meets all criteria, False otherwise

    Raises:
    -------
    FieldNotFoundError
        If field doesn't exist in DataFrame
    """
    if field_name not in df.columns:
        raise FieldNotFoundError(field_name, list(df.columns))

    validator = NumericFieldValidator(
        allow_null=allow_null, min_value=min_value, max_value=max_value
    )

    try:
        result = validator.validate(df[field_name], field_name=field_name)
        return result.is_valid
    except ValidationError:
        return False


def validate_categorical_field(
    df: pd.DataFrame,
    field_name: str,
    allow_null: bool = True,
    min_categories: Optional[int] = None,
    max_categories: Optional[int] = None,
    valid_categories: Optional[List[str]] = None,
    min_frequency_threshold: Optional[int] = None,
    check_distribution: bool = False,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Enhanced validation for categorical fields with distribution analysis.

    DEPRECATED: Use CategoricalFieldValidator instead for more features.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    min_categories : Optional[int]
        Minimum number of unique categories (default: None)
    max_categories : Optional[int]
        Maximum number of unique categories (default: None)
    valid_categories : Optional[List[str]]
        List of valid category values (default: None)
    min_frequency_threshold : Optional[int]
        Minimum frequency for rare category detection (default: None)
    check_distribution : bool, optional
        Whether to analyze category distribution (default: False)
    logger_instance : Optional[logging.Logger]
        Logger instance for logging validation issues

    Returns
    -------
    Tuple[bool, Dict[str, Any]]
        (True if valid, validation details including distribution info)
    """
    # Use provided logger or module logger
    log = logger_instance or logger

    # Check if field exists
    if field_name not in df.columns:
        return False, {
            "valid": False,
            "field_name": field_name,
            "error": f"Field '{field_name}' not found",
            "errors": [f"Field '{field_name}' not found in DataFrame"],
        }

    # Create validator with available parameters
    validator = CategoricalFieldValidator(
        allow_null=allow_null,
        valid_categories=valid_categories,
        max_categories=max_categories,
    )

    try:
        # Validate the field
        result = validator.validate(df[field_name], field_name=field_name)

        # Convert ValidationResult to legacy format with explicit type annotation
        details: Dict[str, Any] = {
            "valid": result.is_valid,
            "field_name": field_name,
            "errors": result.errors,
            "warnings": result.warnings,
            **result.details,
        }

        # Add extended analysis if validation passed
        if result.is_valid:
            series = df[field_name]

            # Check minimum categories constraint
            if min_categories is not None:
                unique_count = series.nunique()
                if unique_count < min_categories:
                    details["valid"] = False
                    details.setdefault("errors", []).append(
                        f"Too few categories: {unique_count} < {min_categories}"
                    )

            # Add distribution analysis if requested
            if check_distribution:
                value_counts = series.value_counts(dropna=False)

                # Convert to dict properly
                details["distribution"] = value_counts.to_dict()

                # Create distribution statistics
                details["distribution_stats"] = {
                    "total_unique": value_counts.size,  # More efficient than len()
                    "most_common": value_counts.head(5).to_dict(),  # Proper conversion
                    "least_common": value_counts.tail(5).to_dict(),  # Proper conversion
                }

                # Check for rare categories if threshold provided
                if min_frequency_threshold is not None:
                    rare_categories = value_counts[
                        value_counts < min_frequency_threshold
                    ]
                    details["rare_categories"] = rare_categories.to_dict()
                    details["rare_category_count"] = len(rare_categories)  # Already int

        return details["valid"], details

    except Exception as e:
        log.error(f"Error validating categorical field '{field_name}': {e}")
        return False, {
            "valid": False,
            "field_name": field_name,
            "error": str(e),
            "errors": [str(e)],
        }


def validate_hierarchy_dictionary(
    hierarchy_dict: Union[Any, Dict[str, Any], str, Path],
    required_levels: Optional[int] = None,
    check_coverage: Optional[List[str]] = None,
    min_coverage: float = 0.8,
    logger_instance: Optional[logging.Logger] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate hierarchy dictionary structure and content.

    DEPRECATED: Use HierarchyFileValidator instead.

    Parameters:
    -----------
    hierarchy_dict : Union[Any, Dict[str, Any], str, Path]
        Hierarchy dictionary to validate (object, dict, or file path)
    required_levels : Optional[int]
        Required number of hierarchy levels (default: None)
    check_coverage : Optional[List[str]]
        List of values to check coverage for (default: None)
    min_coverage : float, optional
        Minimum required coverage ratio (default: 0.8)
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored, uses module logger)

    Returns:
    --------
    Tuple[bool, Dict[str, Any]]
        (True if valid, validation details)
    """
    try:
        if isinstance(hierarchy_dict, (str, Path)):
            # Use file validator for paths
            validator = HierarchyFileValidator(validate_structure=True)
            result = validator.validate(hierarchy_dict)

            # Add additional checks if needed
            if result.is_valid and required_levels is not None:
                # This would need to be implemented in HierarchyFileValidator
                result.warnings.append(
                    f"Required levels check ({required_levels}) not implemented in new validator"
                )

            if result.is_valid and check_coverage is not None:
                # Coverage check would need implementation
                result.warnings.append(
                    f"Coverage check for {len(check_coverage)} values not implemented"
                )

        else:
            # For dict or object, create basic validation result
            result = ValidationResult(
                is_valid=isinstance(hierarchy_dict, dict),
                warnings=[
                    "Direct dictionary validation is limited, consider using HierarchyFileValidator"
                ],
            )

            if result.is_valid:
                result.details["type"] = "dictionary"
                result.details["size"] = len(hierarchy_dict)

                # Basic structure analysis
                if required_levels is not None:
                    # Simple depth check (would need recursive implementation)
                    result.warnings.append(
                        "Hierarchy depth validation not fully implemented for dictionaries"
                    )

        # Convert to legacy format
        details = {
            "valid": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            **result.details,
        }

        return result.is_valid, details

    except Exception as e:
        logger.error(f"Error validating hierarchy dictionary: {e}")
        return False, {"valid": False, "error": str(e), "errors": [str(e)]}


@validation_handler()
def validate_datetime_field(
    df: pd.DataFrame,
    field_name: str,
    allow_null: bool = True,
    min_date: Optional[pd.Timestamp] = None,
    max_date: Optional[pd.Timestamp] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate that a field is a datetime type with optional range checking.

    DEPRECATED: Use DateTimeFieldValidator instead.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the field
    field_name : str
        The name of the field to validate
    allow_null : bool, optional
        Whether to allow null values (default: True)
    min_date : Optional[pd.Timestamp]
        Minimum allowed date (default: None)
    max_date : Optional[pd.Timestamp]
        Maximum allowed date (default: None)
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored, uses module logger)

    Returns:
    --------
    bool
        True if the field is datetime and meets all criteria, False otherwise
    """
    if field_name not in df.columns:
        raise FieldNotFoundError(field_name, list(df.columns))

    validator = DateTimeFieldValidator(
        allow_null=allow_null, min_date=min_date, max_date=max_date
    )

    try:
        result = validator.validate(df[field_name], field_name=field_name)
        return result.is_valid
    except ValidationError:
        return False


# =============================================================================
# Strategy Validation Wrappers (Backward Compatibility)
# =============================================================================


def validate_generalization_strategy(
    strategy: str,
    valid_strategies: List[str],
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate that a generalization strategy is supported.

    DEPRECATED: Use validate_strategy() from new system.
    This is a legacy wrapper for backward compatibility.

    Parameters:
    -----------
    strategy : str
        The strategy to validate
    valid_strategies : List[str]
        List of valid strategies
    logger_instance : Optional[logging.Logger]
        Logger instance (ignored)

    Returns:
    --------
    bool
        True if the strategy is valid, False otherwise
    """
    try:
        result = validate_strategy_new(strategy, valid_strategies)
        return result.is_valid
    except Exception as e:
        logger.error(f"Strategy validation error: {e}")
        return False


# =============================================================================
# Factory Functions (New Convenient API)
# =============================================================================


def create_validator(field_type: str, **params) -> BaseValidator:
    """
    Factory function for creating field validators.

    This is a convenience wrapper that provides a simpler interface
    for creating validators of various types.

    Parameters:
    -----------
    field_type : str
        Type of field validator to create:
        - 'numeric': NumericFieldValidator
        - 'categorical': CategoricalFieldValidator
        - 'datetime': DateTimeFieldValidator
        - 'boolean': BooleanFieldValidator
        - 'text': TextFieldValidator
        - 'network': NetworkValidator
        - 'geographic': GeographicValidator
        - 'temporal': TemporalValidator
        - 'financial': FinancialValidator
        - 'file': FilePathValidator
        - 'directory': DirectoryPathValidator
    **params : dict
        Parameters specific to the validator type

    Returns:
    --------
    BaseValidator
        Configured validator instance

    Examples:
    ---------
    >>> # Create numeric validator
    >>> num_validator = create_validator('numeric', min_value=0, max_value=100)
    >>>
    >>> # Create categorical validator
    >>> cat_validator = create_validator('categorical', valid_categories=['A', 'B', 'C'])
    >>>
    >>> # Create file validator
    >>> file_validator = create_validator('file', must_exist=True, valid_extensions=['.csv'])
    """
    # Map extended types to specialized validators
    specialized_validators = {
        "network": NetworkValidator,
        "geographic": GeographicValidator,
        "temporal": TemporalValidator,
        "financial": FinancialValidator,
        "file": FilePathValidator,
        "directory": DirectoryPathValidator,
        "json": JSONFileValidator,
        "csv": CSVFileValidator,
        "hierarchy": HierarchyFileValidator,
    }

    if field_type in specialized_validators:
        return specialized_validators[field_type](**params)

    # Use standard field validator factory for basic types
    return create_field_validator(field_type, **params)


def create_validation_pipeline(
    *validators: BaseValidator, stop_on_first_error: bool = False
) -> CompositeValidator:
    """
    Create a validation pipeline from multiple validators.

    This is a convenience function for creating composite validators
    with a more intuitive interface.

    Parameters:
    -----------
    *validators : BaseValidator
        Variable number of validators to chain together
    stop_on_first_error : bool
        Whether to stop validation chain on first error

    Returns:
    --------
    CompositeValidator
        Composite validator that runs all validators in sequence
    """
    return CompositeValidator(list(validators), stop_on_first_error=stop_on_first_error)


def validate_dataframe_schema(
    df: pd.DataFrame, schema: Dict[str, Dict[str, Any]], strict: bool = False
) -> ValidationResult:
    """
    Validate entire DataFrame against a schema definition.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    schema : Dict[str, Dict[str, Any]]
        Schema definition where keys are column names and values are
        validator configurations with 'type' and additional parameters
    strict : bool
        If True, unexpected columns cause validation failure

    Returns:
    --------
    ValidationResult
        Aggregated validation result for all columns
    """
    results = []
    errors = []
    warnings = []
    validated_columns = set()

    # Validate each column in schema
    for column, config in schema.items():
        if column not in df.columns:
            errors.append(f"Required column '{column}' not found")
            continue

        # Extract validator type and create validator
        validator_config = config.copy()
        validator_type = validator_config.pop("type", "text")

        try:
            validator = create_validator(validator_type, **validator_config)
            result = validator.validate(df[column])
            result.field_name = column
            results.append(result)
            validated_columns.add(column)

            if not result.is_valid:
                errors.extend([f"{column}: {err}" for err in result.errors])
            warnings.extend([f"{column}: {warn}" for warn in result.warnings])

        except Exception as e:
            errors.append(f"{column}: Validation setup failed - {str(e)}")

    # Check for unexpected columns
    expected_columns = set(schema.keys())
    actual_columns = set(df.columns)
    unexpected = actual_columns - expected_columns

    if unexpected:
        msg = f"Unexpected columns found: {', '.join(sorted(unexpected))}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Create aggregate result
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        details={
            "validated_columns": len(validated_columns),
            "total_columns": len(df.columns),
            "schema_columns": len(schema),
            "column_results": {
                r.field_name: r.is_valid for r in results if r.field_name
            },
            "unexpected_columns": list(unexpected) if unexpected else [],
        },
    )


def create_cross_validator(
    validators: Dict[str, BaseValidator], validation_order: Optional[List[str]] = None
) -> BaseValidator:
    """
    Create a cross-validator that validates multiple aspects in a specific order.

    This is useful for complex validation scenarios where you need to validate
    type, then format, then business rules, etc.

    Parameters:
    -----------
    validators : Dict[str, BaseValidator]
        Dictionary of named validators
    validation_order : Optional[List[str]]
        Order to execute validators (if None, uses dict order)

    Returns:
    --------
    BaseValidator
        Composite validator with ordered execution

    Examples:
    ---------
    >>> # Example 1: Simple cross-validation
    >>> type_validator = create_validator('numeric')
    >>> range_validator = create_validator('numeric', min_value=0, max_value=100)
    >>>
    >>> cross_val = create_cross_validator({
    ...     'type': type_validator,
    ...     'range': range_validator
    ... }, validation_order=['type', 'range'])
    >>>
    >>> # Example 2: With custom validator (assuming you have one)
    >>> # from myapp.validators import CustomBusinessRuleValidator
    >>> # cross_validator = create_cross_validator({
    >>> #     'type': create_validator('numeric'),
    >>> #     'range': create_validator('numeric', min_value=0, max_value=100),
    >>> #     'business': CustomBusinessRuleValidator()
    >>> # }, validation_order=['type', 'range', 'business'])
    """
    if validation_order:
        ordered_validators = [
            validators[name] for name in validation_order if name in validators
        ]
    else:
        ordered_validators = list(validators.values())

    return CompositeValidator(ordered_validators, stop_on_first_error=True)


# =============================================================================
# Specialized Validation Wrappers (Backward Compatibility)
# =============================================================================


def validate_geographic_data(
    data: pd.Series, geo_type: str, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Validate geographic data format.

    DEPRECATED: Use GeographicValidator directly.
    """
    try:
        validator = GeographicValidator(geo_type=geo_type)
        result = validator.validate(data)
        return result.is_valid
    except Exception as e:
        logger.error(f"Geographic validation error: {e}")
        return False


def validate_temporal_sequence(
    data: pd.Series, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Validate temporal sequence data.

    DEPRECATED: Use TemporalValidator directly.
    """
    try:
        validator = TemporalValidator(check_sequence=True)
        result = validator.validate(data)
        return result.is_valid
    except Exception as e:
        logger.error(f"Temporal validation error: {e}")
        return False


def validate_network_identifiers(
    data: pd.Series, network_type: str, logger_instance: Optional[logging.Logger] = None
) -> bool:
    """
    Validate network identifier format.

    DEPRECATED: Use NetworkValidator directly.
    """
    try:
        validator = NetworkValidator(network_type=network_type)
        result = validator.validate(data)
        return result.is_valid
    except Exception as e:
        logger.error(f"Network validation error: {e}")
        return False


def validate_financial_data(
    data: pd.Series,
    financial_type: str,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate financial data format.

    DEPRECATED: Use FinancialValidator directly.
    """
    try:
        validator = FinancialValidator(financial_type=financial_type)
        result = validator.validate(data)
        return result.is_valid
    except Exception as e:
        logger.error(f"Financial validation error: {e}")
        return False


# =============================================================================
# File and Path Validation Wrappers (Backward Compatibility)
# =============================================================================


def validate_file_path(
    file_path: Union[str, Path],
    must_exist: bool = True,
    valid_extensions: Optional[List[str]] = None,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate file path and optionally check existence and extension.

    DEPRECATED: Use FilePathValidator directly.
    """
    try:
        validator = FilePathValidator(
            must_exist=must_exist, valid_extensions=valid_extensions
        )
        result = validator.validate(file_path)
        return result.is_valid
    except Exception as e:
        logger.error(f"File path validation error: {e}")
        return False


def validate_directory_path(
    dir_path: Union[str, Path],
    must_exist: bool = True,
    create_if_missing: bool = False,
    logger_instance: Optional[logging.Logger] = None,
) -> bool:
    """
    Validate directory path and optionally create if missing.

    DEPRECATED: Use DirectoryPathValidator directly.
    """
    try:
        validator = DirectoryPathValidator(
            must_exist=must_exist, create_if_missing=create_if_missing
        )
        result = validator.validate(dir_path)
        return result.is_valid
    except Exception as e:
        logger.error(f"Directory path validation error: {e}")
        return False


# =============================================================================
# Utility Functions (Re-exports and Wrappers)
# =============================================================================


def get_validation_error_result(
    error_message: str,
    field_name: Optional[str] = None,
    error_type: str = "ValidationError",
) -> Dict[str, Any]:
    """
    Create a standardized validation error result.

    DEPRECATED: Use ValidationResult directly.

    Parameters:
    -----------
    error_message : str
        The error message
    field_name : str, optional
        The field name associated with the error
    error_type : str, optional
        The type of error (default: "ValidationError")

    Returns:
    --------
    Dict[str, Any]
        Validation error result with standardized structure
    """
    result = ValidationResult(
        is_valid=False,
        field_name=field_name,
        errors=[error_message],
        details={"error_type": error_type},
    )
    return result.to_dict()


def get_validation_success_result(
    field_name: Optional[str] = None, additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized validation success result.

    DEPRECATED: Use ValidationResult directly.

    Parameters:
    -----------
    field_name : str, optional
        The field name that was validated
    additional_info : Optional[Dict[str, Any]]
        Additional validation information

    Returns:
    --------
    Dict[str, Any]
        Validation success result with standardized structure
    """
    result = ValidationResult(
        is_valid=True, field_name=field_name, details=additional_info or {}
    )
    return result.to_dict()


# Module metadata
__version__ = "3.2.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"
__updated__ = "2025-06-15"

# Define explicit exports - this helps with auto-discovery and documentation
__all__ = [
    # === Core classes and types from new system ===
    "ValidationResult",
    "BaseValidator",
    "CompositeValidator",
    "ValidationContext",
    "ValidationCache",
    # === All Exceptions ===
    "ValidationError",
    "FieldNotFoundError",
    "FieldTypeError",
    "FieldValueError",
    "FileNotFoundValidationError",
    "InvalidStrategyError",
    "InvalidParameterError",
    "ConditionalValidationError",
    "InvalidDataFormatError",
    "RangeValidationError",
    "FileValidationError",
    "InvalidFileFormatError",
    "ValidationErrorInfo",
    "MultipleValidationErrors",
    "ConfigurationError",
    "raise_if_errors",
    # === Field Validators ===
    "NumericFieldValidator",
    "CategoricalFieldValidator",
    "DateTimeFieldValidator",
    "BooleanFieldValidator",
    "TextFieldValidator",
    # === File Validators ===
    "FilePathValidator",
    "DirectoryPathValidator",
    "HierarchyFileValidator",
    "JSONFileValidator",
    "CSVFileValidator",
    "MultiFileValidator",
    # === Type Validators ===
    "NetworkValidator",
    "GeographicValidator",
    "TemporalValidator",
    "FinancialValidator",
    "SpecializedTypeValidator",
    # === Factory functions ===
    "create_validator",
    "create_field_validator",
    "create_validation_pipeline",
    "create_cross_validator",
    "validate_dataframe_schema",
    # === Legacy field validation (backward compatibility) ===
    "validate_field_exists",
    "validate_multiple_fields_exist",
    "validate_numeric_field",
    "validate_categorical_field",
    "validate_hierarchy_dictionary",
    "validate_datetime_field",
    # === Strategy validation ===
    "validate_strategy_new",  # From new system
    "validate_generalization_strategy",  # Legacy wrapper
    "validate_generalization_strategy_new",  # From new system
    "validate_noise_strategy",
    "validate_suppression_strategy",
    "validate_masking_strategy",
    "validate_pseudonymization_strategy",
    # === Mode validation ===
    "validate_operation_mode",
    "validate_null_strategy",
    # === Parameter validation ===
    "validate_bin_count",
    "validate_precision",
    "validate_range_limits",
    "validate_percentiles",
    # === Strategy-specific validation ===
    "validate_noise_parameters",
    "validate_masking_parameters",
    "validate_hierarchy_parameters",
    "validate_strategy_compatibility",
    "validate_output_field_configuration",
    # === Specialized validation (legacy) ===
    "validate_specialized_type",
    "validate_geographic_data",
    "validate_temporal_sequence",
    "validate_network_identifiers",
    "validate_financial_data",
    # === File validation (legacy) ===
    "validate_file_path",
    "validate_directory_path",
    # === Decorators ===
    "validation_handler",
    "standard_validator",
    "validate_types",
    "sanitize_inputs",
    "skip_if_empty",
    "requires_field",
    "requires_fields",
    "aggregate_results",
    "cached_validation",
    # === Utility functions ===
    "check_field_exists",
    "check_multiple_fields_exist",
    "is_numeric_type",
    "is_categorical_type",
    "is_datetime_type",
    "safe_sample",
    "validate_type",
    "validate_range",
    "get_validation_error_result",
    "get_validation_success_result",
    # === Support classes ===
    "LegacyValidationSupport",
    # === Strategy Constants ===
    "GENERALIZATION_STRATEGIES",
    "NOISE_STRATEGIES",
    "SUPPRESSION_STRATEGIES",
    "MASKING_STRATEGIES",
    "PSEUDONYMIZATION_STRATEGIES",
    "OPERATION_MODES",
    "NULL_STRATEGIES",
    # === Pattern Constants ===
    "NETWORK_PATTERNS",
    "GEO_PATTERNS",
    "FINANCIAL_PATTERNS",
    "COMMON_CURRENCIES",
]

# Log module initialization
logger.info(f"Validation facade v{__version__} initialized with {len(__all__)} exports")
