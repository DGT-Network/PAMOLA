"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Validation Exceptions
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
   Custom exception classes for the validation framework. Provides
   structured error handling with consistent interfaces for different
   validation failure scenarios.

Key Features:
   - Base ValidationError with structured error information
   - Specialized exceptions for different validation contexts
   - Rich error messages with field and context information
   - Error aggregation for multiple validation failures
   - Serializable error representations
   - Integration with ValidationResult

Design Principles:
   - Clarity: Clear error messages with actionable information
   - Structure: Consistent error data across exception types
   - Context: Preserve validation context for debugging
   - Integration: Easy conversion to ValidationResult

Usage:
   Raise appropriate exceptions in validators. The validation_handler
   decorator will catch and convert them to ValidationResult objects.

Dependencies:
   - typing - Type hints
   - dataclasses - Structured error data

Changelog:
   1.0.0 - Initial implementation with core exception types
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Import for type hints
from .base import ValidationResult


# =============================================================================
# Base Validation Exception
# =============================================================================

class ValidationError(Exception):
    """
    Base exception for all validation errors.

    Provides structured error information that can be easily
    converted to ValidationResult objects.

    Attributes:
        message: Error message
        field_name: Name of field that failed validation
        error_code: Machine-readable error code
        details: Additional error details
    """

    def __init__(self,
                 message: str,
                 field_name: Optional[str] = None,
                 error_code: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error.

        Args:
            message: Human-readable error message
            field_name: Field associated with error
            error_code: Machine-readable error identifier
            details: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.field_name = field_name
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary representation."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'field_name': self.field_name,
            'error_code': self.error_code,
            'details': self.details
        }

    def to_validation_result(self) -> ValidationResult:
        """Convert exception to ValidationResult."""
        return ValidationResult(
            is_valid=False,
            field_name=self.field_name,
            errors=[self.message],
            details={'error': self.to_dict()}
        )


# =============================================================================
# Field Validation Exceptions
# =============================================================================

class FieldNotFoundError(ValidationError):
    """Raised when expected field is not found in DataFrame."""

    def __init__(self, field_name: str, available_fields: Optional[List[str]] = None):
        message = f"Field '{field_name}' not found"
        if available_fields:
            message += f". Available fields: {', '.join(available_fields[:10])}"
            if len(available_fields) > 10:
                message += f" and {len(available_fields) - 10} more"

        super().__init__(
            message=message,
            field_name=field_name,
            error_code="FIELD_NOT_FOUND",
            details={'available_fields': available_fields}
        )


class FieldTypeError(ValidationError):
    """Raised when field has unexpected type."""

    def __init__(self,
                 field_name: str,
                 expected_type: str,
                 actual_type: str,
                 convertible: bool = False):
        message = (f"Field '{field_name}' has incorrect type. "
                   f"Expected {expected_type}, got {actual_type}")
        if convertible:
            message += " (but can be converted)"

        super().__init__(
            message=message,
            field_name=field_name,
            error_code="FIELD_TYPE_ERROR",
            details={
                'expected_type': expected_type,
                'actual_type': actual_type,
                'convertible': convertible
            }
        )


class FieldValueError(ValidationError):
    """Raised when field contains invalid values."""

    def __init__(self,
                 field_name: str,
                 reason: str,
                 invalid_count: Optional[int] = None,
                 examples: Optional[List[Any]] = None):
        message = f"Field '{field_name}' contains invalid values: {reason}"
        if invalid_count:
            message += f" ({invalid_count} invalid values)"

        super().__init__(
            message=message,
            field_name=field_name,
            error_code="FIELD_VALUE_ERROR",
            details={
                'reason': reason,
                'invalid_count': invalid_count,
                'examples': examples[:5] if examples else None
            }
        )


# =============================================================================
# Strategy Validation Exceptions
# =============================================================================

class InvalidStrategyError(ValidationError):
    """Raised when anonymization strategy is invalid."""

    def __init__(self,
                 strategy: str,
                 valid_strategies: List[str],
                 operation_type: Optional[str] = None):
        message = f"Invalid strategy '{strategy}'"
        if operation_type:
            message += f" for {operation_type} operation"
        message += f". Valid strategies: {', '.join(valid_strategies)}"

        super().__init__(
            message=message,
            error_code="INVALID_STRATEGY",
            details={
                'strategy': strategy,
                'valid_strategies': valid_strategies,
                'operation_type': operation_type
            }
        )


class InvalidParameterError(ValidationError):
    """Raised when operation parameter is invalid."""

    def __init__(self,
                 param_name: str,
                 param_value: Any,
                 reason: str,
                 valid_range: Optional[str] = None):
        message = f"Invalid parameter '{param_name}' = {param_value}: {reason}"
        if valid_range:
            message += f". Valid range: {valid_range}"

        super().__init__(
            message=message,
            error_code="INVALID_PARAMETER",
            details={
                'param_name': param_name,
                'param_value': param_value,
                'reason': reason,
                'valid_range': valid_range
            }
        )


# =============================================================================
# Conditional Validation Exceptions
# =============================================================================

class ConditionalValidationError(ValidationError):
    """Raised when conditional processing parameters are invalid."""

    def __init__(self,
                 condition_field: Optional[str],
                 condition_operator: str,
                 reason: str):
        message = f"Invalid conditional parameters: {reason}"
        if condition_field:
            message = f"Invalid condition on field '{condition_field}': {reason}"

        super().__init__(
            message=message,
            field_name=condition_field,
            error_code="CONDITIONAL_ERROR",
            details={
                'condition_operator': condition_operator,
                'reason': reason
            }
        )


# =============================================================================
# Data Type Validation Exceptions
# =============================================================================

class InvalidDataFormatError(ValidationError):
    """Raised when data doesn't match expected format."""

    def __init__(self,
                 field_name: str,
                 data_type: str,
                 format_description: str,
                 sample_invalid: Optional[List[Any]] = None):
        message = (f"Field '{field_name}' has invalid {data_type} format. "
                   f"Expected: {format_description}")

        super().__init__(
            message=message,
            field_name=field_name,
            error_code=f"INVALID_{data_type.upper()}_FORMAT",
            details={
                'data_type': data_type,
                'format_description': format_description,
                'sample_invalid': sample_invalid[:3] if sample_invalid else None
            }
        )


class RangeValidationError(ValidationError):
    """Raised when values are outside acceptable range."""

    def __init__(self,
                 field_name: str,
                 min_value: Optional[Union[int, float]] = None,
                 max_value: Optional[Union[int, float]] = None,
                 actual_min: Optional[Union[int, float]] = None,
                 actual_max: Optional[Union[int, float]] = None):
        message = f"Field '{field_name}' values outside acceptable range"

        details = {}
        if min_value is not None:
            details['expected_min'] = min_value
        if max_value is not None:
            details['expected_max'] = max_value
        if actual_min is not None:
            details['actual_min'] = actual_min
        if actual_max is not None:
            details['actual_max'] = actual_max

        if min_value is not None and actual_min is not None and actual_min < min_value:
            message += f". Minimum {actual_min} < {min_value}"
        if max_value is not None and actual_max is not None and actual_max > max_value:
            message += f". Maximum {actual_max} > {max_value}"

        super().__init__(
            message=message,
            field_name=field_name,
            error_code="RANGE_ERROR",
            details=details
        )


# =============================================================================
# File and Path Exceptions
# =============================================================================

class FileValidationError(ValidationError):
    """Raised when file validation fails."""

    def __init__(self,
                 file_path: str,
                 reason: str,
                 error_type: str = "FILE_ERROR"):
        message = f"File validation failed for '{file_path}': {reason}"

        super().__init__(
            message=message,
            error_code=error_type,
            details={
                'file_path': file_path,
                'reason': reason
            }
        )


class FileNotFoundError(FileValidationError):
    """Raised when required file doesn't exist."""

    def __init__(self, file_path: str):
        super().__init__(
            file_path=file_path,
            reason="File does not exist",
            error_type="FILE_NOT_FOUND"
        )


class InvalidFileFormatError(FileValidationError):
    """Raised when file has incorrect format/extension."""

    def __init__(self,
                 file_path: str,
                 expected_formats: List[str],
                 actual_format: Optional[str] = None):
        reason = f"Expected formats: {', '.join(expected_formats)}"
        if actual_format:
            reason += f", got: {actual_format}"

        super().__init__(
            file_path=file_path,
            reason=reason,
            error_type="INVALID_FILE_FORMAT"
        )


# =============================================================================
# Aggregated Validation Exceptions
# =============================================================================

@dataclass
class ValidationErrorInfo:
    """Information about a single validation error."""
    field_name: Optional[str]
    error_type: str
    message: str
    details: Dict[str, Any]


class MultipleValidationErrors(ValidationError):
    """
    Raised when multiple validation errors occur.

    Aggregates multiple validation failures into a single
    exception with detailed information about each error.
    """

    def __init__(self, errors: List[Union[ValidationError, ValidationErrorInfo]]):
        self.errors = []

        # Convert all errors to ValidationErrorInfo
        for error in errors:
            if isinstance(error, ValidationError):
                self.errors.append(ValidationErrorInfo(
                    field_name=error.field_name,
                    error_type=error.__class__.__name__,
                    message=error.message,
                    details=error.details
                ))
            else:
                self.errors.append(error)

        # Create summary message
        field_errors = {}
        for error in self.errors:
            field = error.field_name or "general"
            if field not in field_errors:
                field_errors[field] = []
            field_errors[field].append(error.message)

        messages = []
        for field, msgs in field_errors.items():
            if field == "general":
                messages.extend(msgs)
            else:
                messages.append(f"{field}: {'; '.join(msgs)}")

        super().__init__(
            message=f"Multiple validation errors: {' | '.join(messages)}",
            error_code="MULTIPLE_ERRORS",
            details={'errors': [e.__dict__ for e in self.errors]}
        )

    def to_validation_result(self) -> ValidationResult:
        """Convert to ValidationResult with all errors."""
        result = ValidationResult(is_valid=False)

        for error in self.errors:
            result.errors.append(error.message)
            if error.field_name and not result.field_name:
                result.field_name = error.field_name

        result.details['error_count'] = len(self.errors)
        result.details['errors'] = [e.__dict__ for e in self.errors]

        return result


# =============================================================================
# Configuration Exceptions
# =============================================================================

class ConfigurationError(ValidationError):
    """Raised when operation configuration is invalid."""

    def __init__(self,
                 config_param: str,
                 reason: str,
                 suggestion: Optional[str] = None):
        message = f"Invalid configuration '{config_param}': {reason}"
        if suggestion:
            message += f". Suggestion: {suggestion}"

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={
                'config_param': config_param,
                'reason': reason,
                'suggestion': suggestion
            }
        )


# =============================================================================
# Helper Functions
# =============================================================================

def raise_if_errors(errors: List[ValidationError]) -> None:
    """
    Raise MultipleValidationErrors if list is not empty.

    Args:
        errors: List of validation errors

    Raises:
        MultipleValidationErrors: If errors list is not empty
    """
    if errors:
        if len(errors) == 1:
            raise errors[0]
        else:
            raise MultipleValidationErrors(errors)


# Module exports
__all__ = [
    # Base exception
    'ValidationError',

    # Field exceptions
    'FieldNotFoundError',
    'FieldTypeError',
    'FieldValueError',

    # Strategy exceptions
    'InvalidStrategyError',
    'InvalidParameterError',

    # Conditional exceptions
    'ConditionalValidationError',

    # Data type exceptions
    'InvalidDataFormatError',
    'RangeValidationError',

    # File exceptions
    'FileValidationError',
    'FileNotFoundError',
    'InvalidFileFormatError',

    # Aggregated exceptions
    'ValidationErrorInfo',
    'MultipleValidationErrors',

    # Configuration exceptions
    'ConfigurationError',

    # Helper functions
    'raise_if_errors'
]