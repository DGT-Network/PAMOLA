"""Validation exceptions for PAMOLA Core.

This module provides comprehensive validation exceptions organized by concern:
- Base validation classes
- Field validation (fields, columns, types)
- Parameter validation (params, strategies, types)
- File validation (file access, format, existence)
- Format & range validation
- Conditional & rule validation
- Aggregate validation (multiple errors)
- Utility functions
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from pamola_core.errors.base import (
    BasePamolaError,
    _format_field_list,
    auto_exception,
)
from pamola_core.errors.codes.registry import ErrorCode


# =============================================================================
# BASE VALIDATION
# =============================================================================


class ValidationError(BasePamolaError):
    """
    Base validation error with optional field context.

    Used for data validation, parameter validation, and schema validation.
    Supports field-level context for better error messages.

    Attributes
    ----------
        field_name (Optional[str]): Name of the field that failed validation
    """

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        self.field_name = field_name
        super().__init__(
            message=message,
            error_code=error_code or ErrorCode.DATA_VALIDATION_ERROR,
            details=details or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        """Include field_name in structured output."""
        base_dict = super().to_dict()
        base_dict["field_name"] = self.field_name
        return base_dict


# =============================================================================
# FIELD VALIDATION EXCEPTIONS
# =============================================================================


class FieldNotFoundError(ValidationError):
    """
    Field not found in dataset.

    Raised when attempting to access a field/column that doesn't exist.
    Provides list of available fields for user guidance.
    """

    def __init__(
        self,
        field_name: str,
        available_fields: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        available_display = _format_field_list(available_fields)
        context = f" in dataset '{dataset_name}'" if dataset_name else ""

        message = (
            ErrorMessages.format(
                ErrorCode.FIELD_NOT_FOUND,
                field_name=field_name,
                available_fields=available_display,
            )
            + context
        )

        super().__init__(
            message=message,
            field_name=field_name,
            error_code=ErrorCode.FIELD_NOT_FOUND,
            details={
                "available_fields": available_fields,
                "dataset_name": dataset_name,
            },
        )


class FieldTypeError(ValidationError):
    """Field has wrong data type (e.g., string instead of integer)."""

    def __init__(
        self,
        field_name: str,
        expected_type: str,
        actual_type: str,
        convertible: bool = False,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.FIELD_TYPE_ERROR,
                field_name=field_name,
                expected=expected_type,
                actual=actual_type,
            ),
            field_name=field_name,
            error_code=ErrorCode.FIELD_TYPE_ERROR,
            details={
                "expected_type": expected_type,
                "actual_type": actual_type,
                "convertible": convertible,
            },
        )


class FieldValueError(ValidationError):
    """Field contains invalid values (format, range, or constraint violations)."""

    def __init__(
        self,
        field_name: str,
        reason: str,
        invalid_count: Optional[int] = None,
        examples: Optional[List[Any]] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        if examples:
            sample_value = examples[0]
        elif invalid_count:
            sample_value = f"{invalid_count} values"
        else:
            sample_value = "<unknown>"

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.FIELD_VALUE_ERROR,
                field_name=field_name,
                value=sample_value,
                reason=reason,
            ),
            field_name=field_name,
            error_code=ErrorCode.FIELD_VALUE_ERROR,
            details={
                "reason": reason,
                "invalid_count": invalid_count,
                "examples": examples[:5] if examples else None,
            },
        )


class ColumnNotFoundError(BasePamolaError):
    """Required column not found in DataFrame."""

    def __init__(
        self,
        column_name: Union[str, List[str]],
        available_columns: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        if isinstance(column_name, (list, tuple, set)):
            missing_columns = [str(col) for col in column_name]
        else:
            missing_columns = [str(column_name)]

        missing_display = ", ".join(missing_columns)
        available_display = _format_field_list(available_columns)
        context = f" in dataset '{dataset_name}'" if dataset_name else ""

        message = (
            ErrorMessages.format(
                ErrorCode.COLUMN_NOT_FOUND,
                column_name=missing_display,
                available_columns=available_display,
            )
            + context
        )

        super().__init__(
            message=message,
            error_code=ErrorCode.COLUMN_NOT_FOUND,
            details={
                "missing_columns": missing_columns,
                "available_columns": available_columns,
                "dataset_name": dataset_name,
            },
        )


# =============================================================================
# PARAMETER VALIDATION EXCEPTIONS
# =============================================================================


class InvalidParameterError(ValidationError):
    """Invalid parameter value or type passed to operation."""

    def __init__(
        self,
        param_name: str,
        param_value: Any,
        reason: str,
        valid_range: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.PARAM_INVALID,
                param_name=param_name,
                value=param_value,
                constraint=reason,
            ),
            field_name=None,
            error_code=ErrorCode.PARAM_INVALID,
            details={
                "param_name": param_name,
                "param_value": param_value,
                "reason": reason,
                "valid_range": valid_range,
            },
        )


class MissingParameterError(ValidationError):
    """Missing parameter value or type passed to operation."""

    def __init__(self, param_name: str, operation: Optional[str] = None):
        from pamola_core.errors.messages.registry import ErrorMessages

        operation_display = operation or "<unspecified>"

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.PARAM_MISSING,
                param_name=param_name,
                operation=operation_display,
            ),
            field_name=None,
            error_code=ErrorCode.PARAM_MISSING,
            details={"param_name": param_name, "operation": operation},
        )


class TypeValidationError(ValidationError):
    """Parameter has an invalid type."""

    def __init__(
        self,
        param_name: str,
        expected_type: str,
        actual_type: str,
        message: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        if message is None:
            message = ErrorMessages.format(
                ErrorCode.PARAM_TYPE_ERROR,
                param_name=param_name,
                expected_type=expected_type,
                actual_type=actual_type,
            )

        super().__init__(
            message=message,
            field_name=None,
            error_code=ErrorCode.PARAM_TYPE_ERROR,
            details={
                "param_name": param_name,
                "expected_type": expected_type,
                "actual_type": actual_type,
            },
        )


class InvalidStrategyError(ValidationError):
    """Invalid or unsupported strategy specified for operation."""

    def __init__(
        self,
        strategy: str,
        valid_strategies: List[str],
        operation_type: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.STRATEGY_INVALID,
                strategy=strategy,
                operation_type=operation_type or "<unspecified>",
                valid_strategies=", ".join(valid_strategies),
            ),
            field_name=None,
            error_code=ErrorCode.STRATEGY_INVALID,
            details={
                "strategy": strategy,
                "valid_strategies": valid_strategies,
                "operation_type": operation_type,
            },
        )


# =============================================================================
# FILE VALIDATION EXCEPTIONS
# =============================================================================


class FileValidationError(ValidationError):
    """Base class for file-related validation errors."""

    def __init__(
        self,
        file_path: str,
        reason: str,
        error_code: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.FILE_ERROR,
                file_path=file_path,
                reason=reason,
            ),
            field_name=None,
            error_code=error_code or ErrorCode.FILE_ERROR,
            details={"file_path": file_path, "reason": reason},
        )


class PamolaFileNotFoundError(FileValidationError):
    """
    File not found at specified path.

    Note: Named PamolaFileNotFoundError to avoid shadowing Python's built-in FileNotFoundError.
    """

    def __init__(self, file_path: str):
        super().__init__(
            file_path=file_path,
            reason="File does not exist",
            error_code=ErrorCode.FILE_NOT_FOUND,
        )


class InvalidFileFormatError(FileValidationError):
    """File format doesn't match expected format."""

    def __init__(
        self,
        file_path: str,
        expected_formats: List[str],
        actual_format: Optional[str] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        reason = ErrorMessages.format(
            ErrorCode.FILE_FORMAT_INVALID,
            file_path=file_path,
            expected_format=", ".join(expected_formats),
            actual_format=actual_format or "<unknown>",
        )

        super().__init__(
            file_path=file_path,
            reason=reason,
            error_code=ErrorCode.FILE_FORMAT_INVALID,
        )
        self.details.update(
            {
                "expected_formats": expected_formats,
                "actual_format": actual_format,
            }
        )


# =============================================================================
# FORMAT & RANGE VALIDATION EXCEPTIONS
# =============================================================================


class InvalidDataFormatError(ValidationError):
    """Data format doesn't match expected format (e.g., date format, phone format)."""

    def __init__(
        self,
        field_name: str,
        data_type: str,
        format_description: str,
        sample_invalid: Optional[List[Any]] = None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.VALIDATION_FORMAT_INVALID,
                field_name=field_name,
                expected_format=format_description,
                actual_format=data_type,
            ),
            field_name=field_name,
            error_code=ErrorCode.VALIDATION_FORMAT_INVALID,
            details={
                "data_type": data_type,
                "format_description": format_description,
                "sample_invalid": sample_invalid[:3] if sample_invalid else None,
            },
        )


class RangeValidationError(ValidationError):
    """Value falls outside expected range (min/max constraints)."""

    def __init__(
        self,
        field_name: str,
        min_value=None,
        max_value=None,
        actual_min=None,
        actual_max=None,
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        if actual_min is not None and actual_max is not None:
            value_display = f"{actual_min}..{actual_max}"
        elif actual_min is not None:
            value_display = actual_min
        elif actual_max is not None:
            value_display = actual_max
        else:
            value_display = "<unknown>"

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.VALIDATION_RANGE_FAILED,
                field_name=field_name,
                value=value_display,
                min_value=min_value,
                max_value=max_value,
            ),
            field_name=field_name,
            error_code=ErrorCode.VALIDATION_RANGE_FAILED,
            details={
                "expected_min": min_value,
                "expected_max": max_value,
                "actual_min": actual_min,
                "actual_max": actual_max,
            },
        )


# =============================================================================
# CONDITIONAL & RULE VALIDATION EXCEPTIONS
# =============================================================================


class ConditionalValidationError(ValidationError):
    """Conditional validation rule failed (e.g., if X then Y must be Z)."""

    def __init__(
        self, condition_field: Optional[str], condition_operator: str, reason: str
    ):
        from pamola_core.errors.messages.registry import ErrorMessages

        condition_display = (
            f"{condition_operator} ({reason})" if reason else condition_operator
        )

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.VALIDATION_CONDITIONAL_FAILED,
                field_name=condition_field or "<general>",
                condition=condition_display,
                reason=reason,
            ),
            field_name=condition_field,
            error_code=ErrorCode.VALIDATION_CONDITIONAL_FAILED,
            details={"condition_operator": condition_operator, "reason": reason},
        )


@auto_exception(
    default_error_code=ErrorCode.VALIDATION_MARKER_FAILED,
    message_params=["marker_name", "reason"],
    detail_params=["marker_name", "reason"],
)
class MarkerValidationError(BasePamolaError):
    """Raised when marker validation fails."""

    pass


# =============================================================================
# AGGREGATE VALIDATION EXCEPTIONS
# =============================================================================


@dataclass
class ValidationErrorInfo:
    """
    Structured information about a validation error.

    Used by MultipleValidationErrors to aggregate error details.
    """

    field_name: Optional[str]
    error_type: str
    message: str
    details: Dict[str, Any]


class MultipleValidationErrors(ValidationError):
    """
    Aggregate multiple validation errors into a single exception.

    Groups errors by field for better readability and debugging.

    Examples
    --------
        >>> errors = [
        ...     FieldNotFoundError("user_id", ["name", "email"]),
        ...     FieldTypeError("age", "integer", "string"),
        ... ]
        >>> raise MultipleValidationErrors(errors)
    """

    def __init__(self, errors: Sequence[Union[ValidationError, ValidationErrorInfo]]):
        from pamola_core.errors.messages.registry import ErrorMessages

        self.errors: List[ValidationErrorInfo] = []
        for err in errors:
            if isinstance(err, ValidationError):
                self.errors.append(
                    ValidationErrorInfo(
                        field_name=err.field_name,
                        error_type=err.__class__.__name__,
                        message=err.message,
                        details=err.details,
                    )
                )
            else:
                self.errors.append(err)

        grouped = self._group_errors_by_field()
        summary = self._format_error_summary(grouped)

        super().__init__(
            message=ErrorMessages.format(
                ErrorCode.MULTIPLE_ERRORS, error_count=len(self.errors)
            )
            + f"\n{summary}",
            field_name=None,
            error_code=ErrorCode.MULTIPLE_ERRORS,
            details={"errors": [e.__dict__ for e in self.errors]},
        )

    def _group_errors_by_field(self) -> Dict[str, List[str]]:
        """Group error messages by field name."""
        grouped: Dict[str, List[str]] = {}
        for err in self.errors:
            key = err.field_name or "general"
            grouped.setdefault(key, []).append(err.message)
        return grouped

    def _format_error_summary(self, grouped: Dict[str, List[str]]) -> str:
        """Format grouped errors into readable summary."""
        parts: List[str] = []
        for field, messages in sorted(grouped.items()):
            if field == "general":
                parts.extend(f"  - {message}" for message in messages)
            else:
                parts.append(f"  [{field}]:")
                parts.extend(f"    - {message}" for message in messages)
        return "\n".join(parts)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def raise_if_errors(errors: List[ValidationError]) -> None:
    """
    Raise exception if validation errors list is not empty.

    If single error, raises that error directly.
    If multiple errors, raises MultipleValidationErrors.

    Parameters
    ----------
        errors: List of validation errors to check

    Raises
    ------
        ValidationError: If single error in list
        MultipleValidationErrors: If multiple errors in list

    Examples
    --------
        >>> errors = []
        >>> if not field_valid:
        ...     errors.append(FieldNotFoundError("user_id"))
        >>> if not type_valid:
        ...     errors.append(FieldTypeError("age", "int", "str"))
        >>> raise_if_errors(errors)  # Raises MultipleValidationErrors
    """
    if not errors:
        return
    if len(errors) == 1:
        raise errors[0]
    raise MultipleValidationErrors(errors)
