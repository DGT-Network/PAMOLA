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

# =============================================================================
# Base Infrastructure Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.base import (
    ValidationResult,
    BaseValidator,
    CompositeValidator,
    check_field_exists,
    check_multiple_fields_exist,
)

# =============================================================================
# Error Handling Imports
# =============================================================================
from pamola_core.errors.exceptions.validation import (
    ValidationError,
    FieldNotFoundError,
)

# =============================================================================
# Field Validator Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.field_validators import (
    NumericFieldValidator,
    CategoricalFieldValidator,
    DateTimeFieldValidator,
    create_field_validator,
)

# =============================================================================
# File Validator Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.file_validators import (
    FilePathValidator,
    DirectoryPathValidator,
    HierarchyFileValidator,
    JSONFileValidator,
    CSVFileValidator,
)

# =============================================================================
# Strategy Validator Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.strategy_validators import (
    validate_strategy as validate_strategy_new,
)

# =============================================================================
# Type Validator Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.type_validators import (
    NetworkValidator,
    GeographicValidator,
    TemporalValidator,
    FinancialValidator,
)

# =============================================================================
# Decorator Imports
# =============================================================================
from pamola_core.anonymization.commons.validation.decorators import (
    validation_handler,
)

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
    log = logger_instance or logger

    if field_name not in df.columns:
        return False, {
            "valid": False,
            "field_name": field_name,
            "error": f"Field '{field_name}' not found",
            "errors": [f"Field '{field_name}' not found in DataFrame"],
        }

    validator = CategoricalFieldValidator(
        allow_null=allow_null,
        valid_categories=valid_categories,
        max_categories=max_categories,
    )

    try:
        result = validator.validate(df[field_name], field_name=field_name)

        details: Dict[str, Any] = {
            "valid": result.is_valid,
            "field_name": field_name,
            "errors": result.errors,
            "warnings": result.warnings,
            **result.details,
        }

        if result.is_valid:
            series = df[field_name]

            if min_categories is not None:
                unique_count = series.nunique()
                if unique_count < min_categories:
                    details["valid"] = False
                    details.setdefault("errors", []).append(
                        f"Too few categories: {unique_count} < {min_categories}"
                    )

            if check_distribution:
                value_counts = series.value_counts(dropna=False)
                details["distribution"] = value_counts.to_dict()
                details["distribution_stats"] = {
                    "total_unique": value_counts.size,
                    "most_common": value_counts.head(5).to_dict(),
                    "least_common": value_counts.tail(5).to_dict(),
                }

                if min_frequency_threshold is not None:
                    rare_categories = value_counts[
                        value_counts < min_frequency_threshold
                    ]
                    details["rare_categories"] = rare_categories.to_dict()
                    details["rare_category_count"] = len(rare_categories)

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
            validator = HierarchyFileValidator(validate_structure=True)
            result = validator.validate(hierarchy_dict)

            if result.is_valid and required_levels is not None:
                result.warnings.append(
                    f"Required levels check ({required_levels}) not implemented in new validator"
                )
            if result.is_valid and check_coverage is not None:
                result.warnings.append(
                    f"Coverage check for {len(check_coverage)} values not implemented"
                )
        else:
            result = ValidationResult(
                is_valid=isinstance(hierarchy_dict, dict),
                warnings=[
                    "Direct dictionary validation is limited, consider using HierarchyFileValidator"
                ],
            )

            if result.is_valid:
                result.details["type"] = "dictionary"
                result.details["size"] = len(hierarchy_dict)

                if required_levels is not None:
                    result.warnings.append(
                        "Hierarchy depth validation not fully implemented for dictionaries"
                    )

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

    for column, config in schema.items():
        if column not in df.columns:
            errors.append(f"Required column '{column}' not found")
            continue

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

    unexpected = set(df.columns) - set(schema.keys())
    if unexpected:
        msg = f"Unexpected columns found: {', '.join(sorted(unexpected))}"
        if strict:
            errors.append(msg)
        else:
            warnings.append(msg)

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
    >>> type_validator = create_validator('numeric')
    >>> range_validator = create_validator('numeric', min_value=0, max_value=100)
    >>>
    >>> cross_val = create_cross_validator({
    ...     'type': type_validator,
    ...     'range': range_validator
    ... }, validation_order=['type', 'range'])
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
