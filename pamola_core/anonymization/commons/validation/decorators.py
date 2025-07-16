"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Validation Decorators
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Decorators for validation functions providing common functionality
  such as type checking, error handling, and result standardization.
  These decorators reduce boilerplate code and ensure consistent
  behavior across all validators.

Key Features:
  - Exception handling with ValidationError integration
  - Type validation decorators with runtime checking
  - Input sanitization and normalization
  - Result formatting and standardization
  - Conditional validation decorators

Design Principles:
  - DRY: Eliminate repetitive validation code
  - Consistency: Standardize validation behavior
  - Simplicity: Minimal overhead for MVP
  - Integration: Work with ValidationError exceptions

Usage:
  Apply decorators to validation functions to add common
  functionality without cluttering the validation logic.

Dependencies:
  - functools - Decorator utilities
  - typing - Type hints and runtime checking
  - logging - Error logging
  - inspect - Function signature inspection
  - pandas - DataFrame operations

Changelog:
  1.0.0 - Initial MVP implementation
"""

import inspect
import logging
from functools import wraps
from typing import Callable, List, Optional

import pandas as pd

from .base import ValidationResult
# Import validation exceptions and base classes
from .exceptions import (
    ValidationError,
    FieldNotFoundError,
    FieldTypeError,
    MultipleValidationErrors
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Core Validation Decorators
# =============================================================================

def validation_handler(default_field_name: Optional[str] = None,
                       catch_all_exceptions: bool = True):
    """
    Handle validation exceptions and format results consistently.

    Catches ValidationError exceptions and converts them to ValidationResult
    objects. Optionally catches all exceptions for robustness.

    Args:
        default_field_name: Default field name for results
        catch_all_exceptions: Whether to catch all exceptions or just ValidationError

    Returns:
        Decorator function

    Example:
        @validation_handler(default_field_name="email")
        def validate_email(value):
            if not "@" in value:
                raise FieldValueError("email", "Invalid email format")
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ValidationResult:
            try:
                # Execute the validation function
                result = func(*args, **kwargs)

                # Ensure we return a ValidationResult
                if not isinstance(result, ValidationResult):
                    # Try to convert common return types
                    if isinstance(result, bool):
                        return ValidationResult(
                            is_valid=result,
                            field_name=default_field_name
                        )
                    else:
                        return ValidationResult(
                            is_valid=False,
                            field_name=default_field_name,
                            errors=[
                                f"Validation function must return ValidationResult or bool, got {type(result).__name__}"]
                        )

                # Add default field name if not set
                if default_field_name and not result.field_name:
                    result.field_name = default_field_name

                return result

            except ValidationError as e:
                # Convert ValidationError to ValidationResult
                return e.to_validation_result()

            except Exception as e:
                if not catch_all_exceptions:
                    raise

                # Handle unexpected errors
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return ValidationResult(
                    is_valid=False,
                    field_name=default_field_name,
                    errors=[f"Validation error: {str(e)}"]
                )

        return wrapper

    return decorator


# =============================================================================
# Type Validation Decorators
# =============================================================================

def validate_types(**type_hints):
    """
    Validate function argument types at runtime.

    Checks that arguments match expected types before executing
    the function. Raises FieldTypeError for type mismatches.

    Args:
        **type_hints: Parameter names and their expected types

    Returns:
        Decorator function

    Example:
        @validate_types(df=pd.DataFrame, field_name=str, threshold=(int, float, type(None)))
        def validate_field(df, field_name, threshold=None):
            # df is guaranteed to be DataFrame
            # field_name is guaranteed to be str
            # threshold is int, float, or None
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate each type hint
            for param_name, expected_type in type_hints.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]

                    # Skip None if allowed
                    if value is None and (
                            expected_type is type(None) or
                            (isinstance(expected_type, tuple) and type(None) in expected_type)
                    ):
                        continue

                    # Check type
                    if not isinstance(value, expected_type):
                        # Format type names for error message
                        if isinstance(expected_type, tuple):
                            expected_str = " or ".join(t.__name__ for t in expected_type)
                        else:
                            expected_str = expected_type.__name__

                        # Extract field_name if it's in the arguments
                        field_name = bound_args.arguments.get('field_name', param_name)

                        raise FieldTypeError(
                            field_name=str(field_name),
                            expected_type=expected_str,
                            actual_type=type(value).__name__
                        )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Input Sanitization Decorators
# =============================================================================

def sanitize_inputs(strip_strings: bool = True,
                    normalize_none: bool = True):
    """
    Sanitize and normalize input values.

    Performs common input cleaning operations like stripping
    whitespace from strings and normalizing None values.

    Args:
        strip_strings: Strip whitespace from string inputs
        normalize_none: Convert empty strings to None

    Returns:
        Decorator function

    Example:
        @sanitize_inputs(strip_strings=True)
        def validate_text(text: str):
            # text is guaranteed to be stripped
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Sanitize args
            sanitized_args = []
            for arg in args:
                if isinstance(arg, str):
                    if strip_strings:
                        arg = arg.strip()
                    if normalize_none and not arg:
                        arg = None
                sanitized_args.append(arg)

            # Sanitize kwargs
            sanitized_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if strip_strings:
                        value = value.strip()
                    if normalize_none and not value:
                        value = None
                sanitized_kwargs[key] = value

            return func(*sanitized_args, **sanitized_kwargs)

        return wrapper

    return decorator


# =============================================================================
# Conditional Validation Decorators
# =============================================================================

def skip_if_empty(return_valid: bool = True,
                  check_field: Optional[str] = None):
    """
    Skip validation for empty inputs.

    Useful for optional field validation where empty values
    are acceptable and don't need validation.

    Args:
        return_valid: Return valid result for empty inputs
        check_field: Specific field to check in DataFrame

    Returns:
        Decorator function

    Example:
        @skip_if_empty(return_valid=True)
        def validate_optional_field(value):
            # Won't be called if value is None or empty
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ValidationResult:
            # Determine what to check for emptiness
            if check_field and args and isinstance(args[0], pd.DataFrame):
                # Check specific field in DataFrame
                df = args[0]
                if check_field not in df.columns:
                    return ValidationResult(
                        is_valid=False,
                        field_name=check_field,
                        errors=[f"Field '{check_field}' not found"]
                    )
                value = df[check_field]
            elif args:
                # Check first argument
                value = args[0]
            else:
                # No arguments to check
                return func(*args, **kwargs)

            # Check for various empty conditions
            is_empty = (
                    value is None or
                    (isinstance(value, str) and not value.strip()) or
                    (isinstance(value, (list, dict, tuple)) and len(value) == 0) or
                    (isinstance(value, pd.Series) and (len(value) == 0 or value.isna().all())) or
                    (isinstance(value, pd.DataFrame) and value.empty)
            )

            if is_empty:
                return ValidationResult(
                    is_valid=return_valid,
                    field_name=check_field,
                    warnings=["Skipped validation for empty input"]
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


def requires_field(field_name: str,
                   raise_error: bool = True):
    """
    Ensure DataFrame has required field before validation.

    Args:
        field_name: Required field name
        raise_error: Raise FieldNotFoundError if True, else return ValidationResult

    Returns:
        Decorator function

    Example:
        @requires_field("email")
        def validate_email_format(df):
            # df is guaranteed to have 'email' column
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> ValidationResult:
            # Validate DataFrame type
            if not isinstance(df, pd.DataFrame):
                if raise_error:
                    raise FieldTypeError(
                        field_name="df",
                        expected_type="DataFrame",
                        actual_type=type(df).__name__
                    )
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Expected DataFrame, got {type(df).__name__}"]
                )

            # Check field existence
            if field_name not in df.columns:
                if raise_error:
                    raise FieldNotFoundError(
                        field_name=field_name,
                        available_fields=list(df.columns)
                    )
                return ValidationResult(
                    is_valid=False,
                    field_name=field_name,
                    errors=[f"Required field '{field_name}' not found"]
                )

            return func(df, *args, **kwargs)

        return wrapper

    return decorator


def requires_fields(field_names: List[str],
                    all_required: bool = True):
    """
    Ensure DataFrame has required fields before validation.

    Args:
        field_names: List of required field names
        all_required: If True, all fields must exist; if False, at least one

    Returns:
        Decorator function

    Example:
        @requires_fields(["name", "email"], all_required=True)
        def validate_user_data(df):
            # df has both 'name' and 'email' columns
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(df: pd.DataFrame, *args, **kwargs) -> ValidationResult:
            if not isinstance(df, pd.DataFrame):
                raise FieldTypeError(
                    field_name="df",
                    expected_type="DataFrame",
                    actual_type=type(df).__name__
                )

            missing_fields = [f for f in field_names if f not in df.columns]

            if all_required and missing_fields:
                raise MultipleValidationErrors([
                    FieldNotFoundError(field_name=f, available_fields=list(df.columns))
                    for f in missing_fields
                ])
            elif not all_required and len(missing_fields) == len(field_names):
                raise FieldNotFoundError(
                    field_name=field_names[0],
                    available_fields=list(df.columns)
                )

            return func(df, *args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Composite Decorators
# =============================================================================

def standard_validator(field_name: Optional[str] = None,
                       strip_inputs: bool = True,
                       catch_all: bool = True):
    """
    Apply standard validation decorators in correct order.

    Combines common decorators for typical validation functions:
    - Input sanitization
    - Exception handling

    Args:
        field_name: Default field name for results
        strip_inputs: Strip string inputs
        catch_all: Catch all exceptions

    Returns:
        Decorator function

    Example:
        @standard_validator(field_name="username")
        def validate_username(value):
            if len(value) < 3:
                raise FieldValueError("username", "Too short")
            return ValidationResult(is_valid=True)
    """

    def decorator(func: Callable) -> Callable:
        # Apply decorators in correct order (innermost first)
        wrapped = func

        # Input sanitization (innermost)
        if strip_inputs:
            wrapped = sanitize_inputs(strip_strings=True)(wrapped)

        # Exception handling (outermost)
        wrapped = validation_handler(
            default_field_name=field_name,
            catch_all_exceptions=catch_all
        )(wrapped)

        return wrapped

    return decorator


# =============================================================================
# Result Aggregation Decorator
# =============================================================================

def aggregate_results(stop_on_first_error: bool = False):
    """
    Aggregate multiple validation results from a generator function.

    Allows validation functions to yield multiple results which are
    then aggregated into a single result.

    Args:
        stop_on_first_error: Stop validation on first error

    Returns:
        Decorator function

    Example:
        @aggregate_results()
        def validate_multiple_fields(df):
            for field in ['name', 'email', 'phone']:
                if field not in df.columns:
                    yield ValidationResult(False, field, [f"Missing {field}"])
                else:
                    yield ValidationResult(True, field)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> ValidationResult:
            results = []
            errors = []
            warnings = []

            # Collect all results
            for result in func(*args, **kwargs):
                if isinstance(result, ValidationResult):
                    results.append(result)
                    errors.extend(result.errors)
                    warnings.extend(result.warnings)

                    if not result.is_valid and stop_on_first_error:
                        break

            # Aggregate results
            all_valid = all(r.is_valid for r in results)

            # Merge details
            details = {}
            for i, result in enumerate(results):
                if result.details:
                    details[f"validation_{i}"] = result.details

            return ValidationResult(
                is_valid=all_valid,
                errors=errors,
                warnings=warnings,
                details=details
            )

        return wrapper

    return decorator


# Module exports
__all__ = [
    # Core decorators
    'validation_handler',
    'validate_types',

    # Input handling
    'sanitize_inputs',
    'skip_if_empty',
    'requires_field',
    'requires_fields',

    # Composite decorators
    'standard_validator',

    # Result handling
    'aggregate_results'
]