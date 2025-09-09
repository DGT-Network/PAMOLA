"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Field Type Validators
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Validators for different field types including numeric, categorical,
  datetime, boolean, and text fields. Provides comprehensive validation
  for data quality, type consistency, and value constraints.

Key Features:
  - Type-specific validation classes
  - Range and constraint checking
  - Null value handling
  - Pattern validation for text fields
  - Integration with validation exceptions

Design Principles:
  - Single Responsibility: Each validator handles one field type
  - Error handling via exceptions
  - Minimal but complete validation

Dependencies:
  - pandas - DataFrame operations
  - numpy - Numeric operations
  - re - Pattern matching
  - typing - Type hints
"""

import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from .base import BaseValidator, ValidationResult
from .exceptions import (
    FieldTypeError,
    FieldValueError,
    RangeValidationError,
    InvalidDataFormatError,
)


# =============================================================================
# Numeric Field Validator
# =============================================================================


class NumericFieldValidator(BaseValidator):
    """
    Validator for numeric fields with range checks.

    Validates numeric data and optionally checks value ranges.
    """

    def __init__(
        self,
        allow_null: bool = True,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_inf: bool = False,
    ):
        """
        Initialize numeric field validator.

        Args:
            allow_null: Whether null values are allowed
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            allow_inf: Whether infinite values are allowed
        """
        super().__init__()
        self.allow_null = allow_null
        self.min_value = min_value
        self.max_value = max_value
        self.allow_inf = allow_inf

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate numeric field."""
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Type check
        if not pd.api.types.is_numeric_dtype(series):
            raise FieldTypeError(
                field_name=field_name or "field",
                expected_type="numeric",
                actual_type=str(series.dtype),
            )

        # Null check
        null_count = series.isnull().sum()
        if null_count > 0 and not self.allow_null:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {null_count} null values",
                invalid_count=null_count,
            )

        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            result.add_warning("Field contains only null values")
            return result

        # Infinite values check
        inf_count = np.isinf(non_null).sum()
        if inf_count > 0 and not self.allow_inf:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {inf_count} infinite values",
                invalid_count=inf_count,
            )

        # Range validation
        actual_min = float(non_null.min())
        actual_max = float(non_null.max())

        if self.min_value is not None and actual_min < self.min_value:
            raise RangeValidationError(
                field_name=field_name or "field",
                min_value=self.min_value,
                actual_min=actual_min,
            )

        if self.max_value is not None and actual_max > self.max_value:
            raise RangeValidationError(
                field_name=field_name or "field",
                max_value=self.max_value,
                actual_max=actual_max,
            )

        # Add statistics to result
        result.details["statistics"] = {
            "count": int(len(non_null)),
            "mean": float(non_null.mean()),
            "std": float(non_null.std()),
            "min": actual_min,
            "max": actual_max,
        }

        return result


# =============================================================================
# Categorical Field Validator
# =============================================================================


class CategoricalFieldValidator(BaseValidator):
    """
    Validator for categorical fields.

    Validates categorical data and checks for valid categories.
    """

    def __init__(
        self,
        allow_null: bool = True,
        valid_categories: Optional[List[str]] = None,
        max_categories: Optional[int] = None,
    ):
        """
        Initialize categorical field validator.

        Args:
            allow_null: Whether null values are allowed
            valid_categories: List of valid category values
            max_categories: Maximum number of unique categories
        """
        super().__init__()
        self.allow_null = allow_null
        self.valid_categories = set(valid_categories) if valid_categories else None
        self.max_categories = max_categories

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate categorical field."""
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Type check - allow string, category, or object
        valid_types = (
            pd.api.types.is_string_dtype(series)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_object_dtype(series)
        )

        if not valid_types:
            raise FieldTypeError(
                field_name=field_name or "field",
                expected_type="categorical",
                actual_type=str(series.dtype),
            )

        # Null check
        null_count = series.isnull().sum()
        if null_count > 0 and not self.allow_null:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {null_count} null values",
                invalid_count=null_count,
            )

        # Category analysis
        unique_values = series.dropna().unique()
        unique_count = len(unique_values)

        # Cardinality check
        if self.max_categories and unique_count > self.max_categories:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Too many categories: {unique_count} > {self.max_categories}",
                invalid_count=unique_count,
            )

        # Valid categories check
        if self.valid_categories:
            invalid = set(unique_values) - self.valid_categories
            if invalid:
                raise FieldValueError(
                    field_name=field_name or "field",
                    reason="Invalid categories found",
                    invalid_count=len(invalid),
                    examples=list(invalid)[:5],
                )

        # Add category info to result
        result.details["unique_count"] = unique_count
        result.details["top_categories"] = (
            pd.Series(unique_values).value_counts().head(10).to_dict()
        )

        return result


# =============================================================================
# DateTime Field Validator
# =============================================================================


class DateTimeFieldValidator(BaseValidator):
    """
    Validator for datetime fields.

    Validates datetime data and checks for valid ranges.
    """

    def __init__(
        self,
        allow_null: bool = True,
        min_date: Optional[Union[str, pd.Timestamp]] = None,
        max_date: Optional[Union[str, pd.Timestamp]] = None,
        future_dates_allowed: bool = True,
    ):
        """
        Initialize datetime field validator.

        Args:
            allow_null: Whether null values are allowed
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            future_dates_allowed: Whether future dates are valid
        """
        super().__init__()
        self.allow_null = allow_null
        self.min_date = pd.Timestamp(min_date) if min_date else None
        self.max_date = pd.Timestamp(max_date) if max_date else None
        self.future_dates_allowed = future_dates_allowed

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate datetime field."""
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Try to convert if not already datetime
        if not pd.api.types.is_datetime64_any_dtype(series):
            try:
                series = pd.to_datetime(series, errors="coerce")
                series = pd.Series(series.values, index=series.index)
            except Exception:
                raise FieldTypeError(
                    field_name=field_name or "field",
                    expected_type="datetime",
                    actual_type=str(series.dtype),
                )

        # Null check
        null_count = series.isnull().sum()
        if null_count > 0 and not self.allow_null:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {null_count} null values",
                invalid_count=null_count,
            )

        # Get non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            result.add_warning("Field contains only null values")
            return result

        # Range checks
        actual_min = non_null.min()
        actual_max = non_null.max()

        if self.min_date and actual_min < self.min_date:
            raise RangeValidationError(
                field_name=field_name or "field",
                min_value=self.min_date,
                actual_min=actual_min,
            )

        if self.max_date and actual_max > self.max_date:
            raise RangeValidationError(
                field_name=field_name or "field",
                max_value=self.max_date,
                actual_max=actual_max,
            )

        # Future dates check
        if not self.future_dates_allowed:
            now = pd.Timestamp.now()
            future_count = (non_null > now).sum()
            if future_count > 0:
                raise FieldValueError(
                    field_name=field_name or "field",
                    reason=f"Contains {future_count} future dates",
                    invalid_count=future_count,
                )

        # Add date range to result
        result.details["date_range"] = {
            "min": str(actual_min),
            "max": str(actual_max),
            "span_days": int((actual_max - actual_min).days),
        }

        return result


# =============================================================================
# Boolean Field Validator
# =============================================================================


class BooleanFieldValidator(BaseValidator):
    """
    Validator for boolean fields.

    Validates boolean data including various representations.
    """

    def __init__(self, allow_null: bool = True):
        """
        Initialize boolean field validator.

        Args:
            allow_null: Whether null values are allowed
        """
        super().__init__()
        self.allow_null = allow_null

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate boolean field."""
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Check if already boolean
        if pd.api.types.is_bool_dtype(series):
            # Just check nulls
            null_count = series.isnull().sum()
            if null_count > 0 and not self.allow_null:
                raise FieldValueError(
                    field_name=field_name or "field",
                    reason=f"Contains {null_count} null values",
                    invalid_count=null_count,
                )
        else:
            # Check for boolean-like values
            unique_values = set(series.dropna().unique())

            # Standard boolean representations
            bool_values = {
                True,
                False,
                1,
                0,
                "1",
                "0",
                "true",
                "false",
                "True",
                "False",
                "yes",
                "no",
                "Yes",
                "No",
            }

            # Check if all values are boolean-like
            non_bool = unique_values - bool_values
            if non_bool:
                raise FieldTypeError(
                    field_name=field_name or "field",
                    expected_type="boolean",
                    actual_type="mixed",
                    convertible=False,
                )

        # Add value counts to result
        result.details["value_counts"] = series.value_counts().to_dict()

        return result


# =============================================================================
# Text Field Validator
# =============================================================================


class TextFieldValidator(BaseValidator):
    """
    Validator for text fields.

    Validates text data with length and pattern checks.
    """

    def __init__(
        self,
        allow_null: bool = True,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
    ):
        """
        Initialize text field validator.

        Args:
            allow_null: Whether null values are allowed
            min_length: Minimum string length
            max_length: Maximum string length
            pattern: Regex pattern to match
        """
        super().__init__()
        self.allow_null = allow_null
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern) if pattern else None

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """Validate text field."""
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Type check
        if not (
            pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
        ):
            raise FieldTypeError(
                field_name=field_name or "field",
                expected_type="text",
                actual_type=str(series.dtype),
            )

        # Null check
        null_count = series.isnull().sum()
        if null_count > 0 and not self.allow_null:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {null_count} null values",
                invalid_count=null_count,
            )

        # Get non-null string values
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            result.add_warning("Field contains only null values")
            return result

        # Length checks
        lengths = non_null.str.len()

        if self.min_length and (lengths < self.min_length).any():
            short_count = (lengths < self.min_length).sum()
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"{short_count} values shorter than {self.min_length}",
                invalid_count=short_count,
            )

        if self.max_length and (lengths > self.max_length).any():
            long_count = (lengths > self.max_length).sum()
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"{long_count} values longer than {self.max_length}",
                invalid_count=long_count,
            )

        # Pattern check
        if self.pattern:
            # Sample for performance
            sample_size = min(1000, len(non_null))
            sample = non_null.sample(n=sample_size, random_state=42)

            non_matching = sample.apply(lambda x: not bool(self.pattern.match(x))).sum()
            if non_matching > 0:
                est_total = int(non_matching * len(non_null) / sample_size)
                raise InvalidDataFormatError(
                    field_name=field_name or "field",
                    data_type="text",
                    format_description=f"Pattern: {self.pattern.pattern}",
                    sample_invalid=sample[
                        ~sample.apply(lambda x: bool(self.pattern.match(x)))
                    ]
                    .head(3)
                    .tolist(),
                )

        # Add length stats to result
        result.details["length_stats"] = {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "mean": float(lengths.mean()),
        }

        return result


# =============================================================================
# Field Validator
# =============================================================================
class FieldExistsValidator(BaseValidator):
    """
    Validator to check if a field exists in a DataFrame.

    Returns a ValidationResult with is_valid=True if the field exists,
    otherwise raises FieldTypeError.
    """

    def validate(self, df: pd.DataFrame, field_name: str) -> ValidationResult:
        """
        Validate that the field exists in the DataFrame.

        Args:
            df: The DataFrame to check.
            field_name: The field/column name to check.

        Returns:
            ValidationResult

        Raises:
            FieldTypeError: If the field does not exist.
        """
        result = ValidationResult(is_valid=True, field_name=field_name)
        if field_name not in df.columns:
            raise FieldTypeError(
                field_name=field_name,
                expected_type="existing column",
                actual_type="missing",
            )
        return result


# =============================================================================
# Pattern Validator
# =============================================================================
class PatternValidator(BaseValidator):
    """
    Validator to check if all values in a Series match a given regex pattern.

    Returns a ValidationResult with is_valid=True if all values match,
    otherwise raises InvalidDataFormatError.
    """

    def __init__(self, pattern: str, allow_null: bool = True):
        """
        Initialize pattern validator.

        Args:
            pattern: Regex pattern to match.
            allow_null: Whether null values are allowed.
        """
        super().__init__()
        self.pattern = re.compile(pattern)
        self.allow_null = allow_null

    def validate(
        self, series: pd.Series, field_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate that all values in the Series match the pattern.

        Args:
            series: The Series to check.
            field_name: Optional field name for error reporting.

        Returns:
            ValidationResult

        Raises:
            InvalidDataFormatError: If any value does not match the pattern.
        """
        result = ValidationResult(is_valid=True, field_name=field_name)

        # Null check
        null_count = series.isnull().sum()
        if null_count > 0 and not self.allow_null:
            raise FieldValueError(
                field_name=field_name or "field",
                reason=f"Contains {null_count} null values",
                invalid_count=null_count,
            )

        # Check pattern
        non_null = series.dropna().astype(str)
        if len(non_null) == 0:
            result.add_warning("Field contains only null values")
            return result

        non_matching = non_null[~non_null.apply(lambda x: bool(self.pattern.match(x)))]
        if not non_matching.empty:
            raise InvalidDataFormatError(
                field_name=field_name or "field",
                data_type="pattern",
                format_description=f"Pattern: {self.pattern.pattern}",
                sample_invalid=non_matching.head(3).tolist(),
            )

        return result


# =============================================================================
# Factory Function
# =============================================================================


def create_field_validator(field_type: str, **kwargs) -> BaseValidator:
    """
    Factory function to create field validators.

    Args:
        field_type: Type of field validator
        **kwargs: Arguments for the validator

    Returns:
        Configured field validator

    Raises:
        ValueError: If field_type is not recognized
    """
    validators = {
        "numeric": NumericFieldValidator,
        "categorical": CategoricalFieldValidator,
        "datetime": DateTimeFieldValidator,
        "boolean": BooleanFieldValidator,
        "text": TextFieldValidator,
    }

    if field_type not in validators:
        raise ValueError(
            f"Unknown field type: {field_type}. "
            f"Valid types: {list(validators.keys())}"
        )

    return validators[field_type](**kwargs)


# Module exports
__all__ = [
    "NumericFieldValidator",
    "CategoricalFieldValidator",
    "DateTimeFieldValidator",
    "BooleanFieldValidator",
    "TextFieldValidator",
    "FieldExistsValidator",
    "PatternValidator",
    "create_field_validator",
]
