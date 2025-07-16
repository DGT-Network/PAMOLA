"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Base Validation Utilities
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
   Base classes and utilities for the validation framework. Provides
   foundational components for all validation operations including
   result structures, base validators, and common utilities.

Key Features:
   - Standardized ValidationResult dataclass
   - Abstract base validator class
   - Validation context for logger management
   - Simple validation cache with TTL
   - Common validation utilities
   - Composable validators

Design Principles:
   - Consistency: All validators return ValidationResult
   - Composability: Validators can be chained and combined
   - Simplicity: Minimal dependencies and clear interfaces
   - Performance: Optional caching for expensive validations

Dependencies:
   - dataclasses - Result structures
   - abc - Abstract base classes
   - typing - Type hints
   - logging - Error reporting
   - time - Cache TTL management
   - functools - Caching utilities
   - pandas - Data validation

Changelog:
   1.0.0 - Initial implementation with core validation infrastructure
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import pandas as pd

# Import exceptions from separate module
if TYPE_CHECKING:
    pass

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Validation Result Structure
# =============================================================================

@dataclass
class ValidationResult:
    """
    Standardized result structure for all validation operations.

    Provides consistent interface for validation results across all
    validator types, supporting errors, warnings, and detailed metadata.

    Attributes:
        is_valid: Whether validation passed
        field_name: Name of validated field (if applicable)
        errors: List of error messages
        warnings: List of warning messages
        details: Additional validation details
        metadata: Optional metadata (timing, cache info, etc.)
    """
    is_valid: bool
    field_name: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:
        """Add error message and mark as invalid."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add warning message without affecting validity."""
        self.warnings.append(message)

    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """
        Merge another validation result into this one.

        Args:
            other: Another ValidationResult to merge

        Returns:
            Self for chaining
        """
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.details.update(other.details)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'is_valid': self.is_valid,
            'field_name': self.field_name,
            'errors': self.errors,
            'warnings': self.warnings,
            'details': self.details,
            'metadata': self.metadata
        }


# =============================================================================
# Base Validator Classes
# =============================================================================

class BaseValidator(ABC):
    """
    Abstract base class for all validators.

    Provides common interface and utilities for validation operations.
    Subclasses must implement the validate method.
    """

    def __init__(self, stop_on_error: bool = False):
        """
        Initialize base validator.

        Args:
            stop_on_error: Whether to stop validation chain on error
        """
        self.stop_on_error = stop_on_error
        self._logger = logger

    @abstractmethod
    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """
        Perform validation on data.

        Args:
            data: Data to validate
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        pass

    def __call__(self, data: Any, **kwargs) -> ValidationResult:
        """Allow validator to be called directly."""
        return self.validate(data, **kwargs)


class CompositeValidator(BaseValidator):
    """
    Validator that combines multiple validators.

    Executes validators in sequence and merges results.
    """

    def __init__(self, validators: List[BaseValidator],
                 stop_on_first_error: bool = False):
        """
        Initialize composite validator.

        Args:
            validators: List of validators to execute
            stop_on_first_error: Stop chain on first error
        """
        super().__init__(stop_on_error=stop_on_first_error)
        self.validators = validators

    def validate(self, data: Any, **kwargs) -> ValidationResult:
        """Execute all validators and merge results."""
        result = ValidationResult(is_valid=True)

        for validator in self.validators:
            val_result = validator.validate(data, **kwargs)
            result.merge(val_result)

            if not val_result.is_valid and self.stop_on_error:
                break

        return result

    def add(self, validator: BaseValidator) -> 'CompositeValidator':
        """Add validator to chain."""
        self.validators.append(validator)
        return self


# =============================================================================
# Validation Context
# =============================================================================

class ValidationContext:
    """
    Context manager for validation operations.

    Manages logger instances and shared validation state.
    """

    def __init__(self, logger_instance: Optional[logging.Logger] = None,
                 cache_enabled: bool = True):
        """
        Initialize validation context.

        Args:
            logger_instance: Logger to use (defaults to module logger)
            cache_enabled: Whether to enable validation caching
        """
        self.logger = logger_instance or logger
        self.cache_enabled = cache_enabled
        self._cache = {} if cache_enabled else None

    def log_error(self, message: str, field: Optional[str] = None) -> None:
        """Log error with optional field context."""
        if field:
            self.logger.error(f"[{field}] {message}")
        else:
            self.logger.error(message)

    def log_warning(self, message: str, field: Optional[str] = None) -> None:
        """Log warning with optional field context."""
        if field:
            self.logger.warning(f"[{field}] {message}")
        else:
            self.logger.warning(message)


# =============================================================================
# Validation Cache
# =============================================================================

class ValidationCache:
    """
    Simple TTL-based cache for validation results.

    Caches expensive validation operations with time-based expiration.
    """

    def __init__(self, ttl: int = 300, max_size: int = 1000):
        """
        Initialize validation cache.

        Args:
            ttl: Time to live in seconds
            max_size: Maximum cache entries
        """
        self.ttl = ttl
        self.max_size = max_size
        self._cache: Dict[str, Tuple[ValidationResult, float]] = {}

    def get(self, key: str) -> Optional[ValidationResult]:
        """Get cached result if valid."""
        if key not in self._cache:
            return None

        result, timestamp = self._cache[key]
        if time.time() - timestamp > self.ttl:
            del self._cache[key]
            return None

        return result

    def set(self, key: str, result: ValidationResult) -> None:
        """Cache validation result."""
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(),
                             key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]

        self._cache[key] = (result, time.time())

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()


# =============================================================================
# Common Validation Utilities
# =============================================================================

def check_field_exists(df: pd.DataFrame, field_name: str) -> bool:
    """
    Check if field exists in DataFrame.

    Args:
        df: DataFrame to check
        field_name: Field name to verify

    Returns:
        True if field exists
    """
    return field_name in df.columns


def check_multiple_fields_exist(df: pd.DataFrame,
                                field_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if multiple fields exist in DataFrame.

    Args:
        df: DataFrame to check
        field_names: List of field names to verify

    Returns:
        Tuple of (all exist, list of missing fields)
    """
    missing = [f for f in field_names if f not in df.columns]
    return len(missing) == 0, missing


def is_numeric_type(series: pd.Series) -> bool:
    """Check if series is numeric type."""
    return pd.api.types.is_numeric_dtype(series)


def is_categorical_type(series: pd.Series) -> bool:
    """Check if series is categorical or string type."""
    return (isinstance(series.dtype, pd.CategoricalDtype) or
            pd.api.types.is_string_dtype(series) or
            pd.api.types.is_object_dtype(series))


def is_datetime_type(series: pd.Series) -> bool:
    """Check if series is datetime type."""
    return pd.api.types.is_datetime64_dtype(series)


def safe_sample(data: pd.Series, sample_size: int = 100) -> pd.Series:
    """
    Safely sample from series for validation.

    Args:
        data: Series to sample from
        sample_size: Maximum sample size

    Returns:
        Sampled series
    """
    non_null = data.dropna()
    if len(non_null) == 0:
        return pd.Series([])

    n_samples = min(sample_size, len(non_null))
    return non_null.sample(n=n_samples, random_state=42)


# =============================================================================
# Validation Decorators
# =============================================================================

def cached_validation(cache_key_func: Optional[Callable] = None):
    """
    Decorator to cache validation results.

    Args:
        cache_key_func: Function to generate cache key from args
    """
    def decorator(func):
        # Create cache instance per decorated function
        cache = ValidationCache()

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if cache_key_func:
                key = cache_key_func(*args, **kwargs)
            else:
                # Simple default: function name + first arg string
                key = f"{func.__name__}_{str(args[0])[:50]}"

            # Check cache
            cached_result = cache.get(key)
            if cached_result is not None:
                cached_result.metadata['from_cache'] = True
                return cached_result

            # Execute validation
            result = func(*args, **kwargs)

            # Cache result
            cache.set(key, result)
            result.metadata['cached_at'] = time.time()

            return result

        # Expose cache for manual management
        wrapper.cache = cache
        return wrapper

    return decorator


def validation_handler(func):
    """
    Decorator to handle validation exceptions and logging.

    Wraps validation functions to provide consistent error handling
    and result formatting. Catches ValidationError from exceptions module.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> ValidationResult:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if it's a ValidationError from exceptions module
            if e.__class__.__name__ == 'ValidationError':
                # Try to use to_validation_result if available
                if hasattr(e, 'to_validation_result'):
                    return e.to_validation_result()
                else:
                    return ValidationResult(
                        is_valid=False,
                        errors=[str(e)],
                        field_name=getattr(e, 'field_name', None),
                        details=getattr(e, 'details', {})
                    )
            else:
                # Unexpected error
                logger.error(f"Unexpected error in {func.__name__}: {e}")
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Internal validation error: {str(e)}"]
                )

    return wrapper


# =============================================================================
# Type Checking Utilities
# =============================================================================

def validate_type(value: Any, expected_types: Union[type, Tuple[type, ...]],
                  param_name: str) -> None:
    """
    Validate parameter type.

    Args:
        value: Value to check
        expected_types: Expected type(s)
        param_name: Parameter name for error message

    Raises:
        TypeError: If type doesn't match
    """
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = " or ".join(t.__name__ for t in expected_types)
        else:
            type_names = expected_types.__name__

        raise TypeError(
            f"{param_name} must be {type_names}, "
            f"got {type(value).__name__}"
        )


def validate_range(value: Union[int, float], min_val: Optional[float] = None,
                   max_val: Optional[float] = None, param_name: str = "value") -> None:
    """
    Validate numeric value is within range.

    Args:
        value: Value to check
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        param_name: Parameter name for error message

    Raises:
        ValueError: If value out of range
    """
    if min_val is not None and value < min_val:
        raise ValueError(f"{param_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{param_name} must be <= {max_val}, got {value}")


# Module exports
__all__ = [
    # Core classes
    'ValidationResult',
    'BaseValidator',
    'CompositeValidator',
    'ValidationContext',
    'ValidationCache',

    # Decorators
    'cached_validation',
    'validation_handler',

    # Utilities
    'check_field_exists',
    'check_multiple_fields_exist',
    'is_numeric_type',
    'is_categorical_type',
    'is_datetime_type',
    'safe_sample',
    'validate_type',
    'validate_range'
]