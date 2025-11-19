"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Validation Rules Framework
Package:       pamola_core.metrics.commons.validation_rules
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Provides a modular and extensible framework for applying validation rules
  to data quality metrics. Supports both system defaults and user-defined
  custom rules with pluggable architecture.

Key Features:
  - Abstract base class for validation rules
  - System default rules (Required, Unique, Datatype checks)
  - Custom rules (Min/Max, Valid values, Regex patterns)
  - Extensible architecture for adding new rule types
  - Integration with existing quality scoring system

Dependencies:
  - pandas - DataFrame operations
  - numpy - statistical calculations
  - re - regex pattern matching
  - abc - abstract base classes
  - typing - type hints
"""

import re
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from pamola_core.common.helpers.data_helper import DataHelper

logger = logging.getLogger(__name__)


class FormatValidator(ABC):
    """Abstract base class for format validators."""

    @abstractmethod
    def validate(self, value: str) -> bool:
        """Validate a string value against the format."""
        pass

    @abstractmethod
    def get_pattern(self) -> str:
        """Get the regex pattern used for validation."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get description of the format."""
        pass


class EmailValidator(FormatValidator):
    """Validator for email format."""

    def __init__(self):
        self.pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    def get_description(self) -> str:
        return "Valid email address format"


class PhoneValidator(FormatValidator):
    """Validator for phone number format."""

    def __init__(self):
        self.pattern = re.compile(r"^\+?[0-9\-\(\)\s]{7,20}$")

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^\+?[0-9\-\(\)\s]{7,20}$"

    def get_description(self) -> str:
        return "Valid phone number format"


class URLValidator(FormatValidator):
    """Validator for URL format."""

    def __init__(self):
        self.pattern = re.compile(
            r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$"
        )

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^https?://(?:[-\w.])+(?:\:[0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:\#(?:[\w.])*)?)?$"

    def get_description(self) -> str:
        return "Valid URL format"


class IPValidator(FormatValidator):
    """Validator for IP address format (IPv4 and IPv6)."""

    def __init__(self):
        # IPv4 pattern
        self.ipv4_pattern = re.compile(
            r"^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
        )
        # IPv6 pattern (simplified)
        self.ipv6_pattern = re.compile(
            r"^(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}$|^::1$|^::$"
        )

    def validate(self, value: str) -> bool:
        return bool(self.ipv4_pattern.match(value) or self.ipv6_pattern.match(value))

    def get_pattern(self) -> str:
        return "IPv4 or IPv6 address pattern"

    def get_description(self) -> str:
        return "Valid IP address format (IPv4 or IPv6)"


class CreditCardValidator(FormatValidator):
    """Validator for credit card format."""

    def __init__(self):
        # Basic credit card pattern (13-19 digits with optional spaces/dashes)
        self.pattern = re.compile(
            r"^[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{1,4}$"
        )

    def validate(self, value: str) -> bool:
        # Remove spaces and dashes for validation
        clean_value = re.sub(r"[\s\-]", "", value)
        return bool(self.pattern.match(value)) and 13 <= len(clean_value) <= 19

    def get_pattern(self) -> str:
        return r"^[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{1,4}$"

    def get_description(self) -> str:
        return "Valid credit card format"


class PostalCodeValidator(FormatValidator):
    """Validator for postal code format (US ZIP codes)."""

    def __init__(self):
        # US ZIP code pattern (5 digits or 5+4 format)
        self.pattern = re.compile(r"^\d{5}(-\d{4})?$")

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^\d{5}(-\d{4})?$"

    def get_description(self) -> str:
        return "Valid US postal code format (ZIP code)"


class SSNValidator(FormatValidator):
    """Validator for Social Security Number format."""

    def __init__(self):
        # SSN pattern (XXX-XX-XXXX)
        self.pattern = re.compile(r"^\d{3}-\d{2}-\d{4}$")

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^\d{3}-\d{2}-\d{4}$"

    def get_description(self) -> str:
        return "Valid SSN format (XXX-XX-XXXX)"


class UUIDValidator(FormatValidator):
    """Validator for UUID format."""

    def __init__(self):
        # UUID pattern (8-4-4-4-12 hexadecimal digits)
        self.pattern = re.compile(
            r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
        )

    def validate(self, value: str) -> bool:
        return bool(self.pattern.match(value))

    def get_pattern(self) -> str:
        return r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

    def get_description(self) -> str:
        return "Valid UUID format"


class RuleType(Enum):
    """Enumeration of validation rule types."""

    REQUIRED = "required"
    UNIQUE = "unique"
    DATATYPE = "datatype"
    FORMAT = "format"
    MIN_MAX = "min_max"
    VALID_VALUES = "valid_values"
    REGEX = "regex"
    CUSTOM = "custom"


class RuleCode(Enum):
    """Stable API enum codes for FE↔BE rule definition."""

    REQUIRED = "REQUIRED"
    UNIQUE = "UNIQUE"

    DATATYPE_INT = "DATATYPE_INT"
    DATATYPE_FLOAT = "DATATYPE_FLOAT"
    DATATYPE_DATE = "DATATYPE_DATE"
    DATATYPE_BOOL = "DATATYPE_BOOL"

    FORMAT_EMAIL = "FORMAT_EMAIL"
    FORMAT_PHONE = "FORMAT_PHONE"
    FORMAT_URL = "FORMAT_URL"
    FORMAT_IP = "FORMAT_IP"
    FORMAT_CREDIT_CARD = "FORMAT_CREDIT_CARD"
    FORMAT_POSTAL_CODE = "FORMAT_POSTAL_CODE"
    FORMAT_SSN = "FORMAT_SSN"
    FORMAT_UUID = "FORMAT_UUID"

    MIN_MAX = "MIN_MAX"
    VALID_VALUES = "VALID_VALUES"
    REGEX = "REGEX"

    @classmethod
    def has_value(cls, value: str) -> bool:
        try:
            cls[value]
            return True
        except KeyError:
            return False


def create_rule_from_code(
    code: str, metadata: Dict[str, Any]
) -> Optional["ValidationRule"]:
    """Factory to build a rule instance from a RuleCode and metadata.

    Parameters
    ----------
    code : str
        One of RuleCode values.
    metadata : Dict[str, Any]
        Additional parameters; used for MIN_MAX (min/max), VALID_VALUES (values), REGEX (pattern).
    """
    try:
        rc = RuleCode[code]
    except KeyError:
        return None

    if rc == RuleCode.REQUIRED:
        return RequiredRule()
    if rc == RuleCode.UNIQUE:
        return UniqueRule()

    if rc == RuleCode.DATATYPE_INT:
        return DatatypeRule("int")
    if rc == RuleCode.DATATYPE_FLOAT:
        return DatatypeRule("float")
    if rc == RuleCode.DATATYPE_DATE:
        return DatatypeRule("date")
    if rc == RuleCode.DATATYPE_BOOL:
        return DatatypeRule("bool")

    if rc == RuleCode.FORMAT_EMAIL:
        return FormatRule("email")
    if rc == RuleCode.FORMAT_PHONE:
        return FormatRule("phone")
    if rc == RuleCode.FORMAT_URL:
        return FormatRule("url")
    if rc == RuleCode.FORMAT_IP:
        return FormatRule("ip")
    if rc == RuleCode.FORMAT_CREDIT_CARD:
        return FormatRule("credit_card")
    if rc == RuleCode.FORMAT_POSTAL_CODE:
        return FormatRule("postal_code")
    if rc == RuleCode.FORMAT_SSN:
        return FormatRule("ssn")
    if rc == RuleCode.FORMAT_UUID:
        return FormatRule("uuid")

    if rc == RuleCode.MIN_MAX:
        return MinMaxRule(min_value=metadata.get("min"), max_value=metadata.get("max"))
    if rc == RuleCode.VALID_VALUES:
        values = metadata.get("values") or metadata.get("valid_values") or []
        return ValidValuesRule(valid_values=list(values))
    if rc == RuleCode.REGEX:
        pattern = metadata.get("pattern")
        if pattern:
            return RegexRule(pattern=pattern)
        return None

    return None


@dataclass
class ValidationResult:
    """Result of applying a validation rule to a data series."""

    is_valid: bool
    error_count: int
    error_indices: List[int] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)
    rule_name: str = ""

    def __post_init__(self):
        """Validate the result data."""
        if self.error_count != len(self.error_indices):
            self.error_count = len(self.error_indices)


class ValidationRule(ABC):
    """Abstract base class for all validation rules."""

    def __init__(
        self,
        rule_name: str,
        enabled: bool = True,
        rule_code: Optional["RuleCode"] = None,
    ):
        """
        Initialize validation rule.

        Parameters
        ----------
        rule_name : str
            Human-readable name for the rule
        enabled : bool, default True
            Whether the rule is enabled
        """
        self.rule_name = rule_name
        self.enabled = enabled
        self.rule_type = RuleType.CUSTOM
        self.rule_code = rule_code

    @abstractmethod
    def validate(self, series: pd.Series) -> ValidationResult:
        """
        Validate a pandas Series against this rule.

        Parameters
        ----------
        series : pd.Series
            Data series to validate

        Returns
        -------
        ValidationResult
            Result of validation including error details
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get human-readable description of the rule.

        Returns
        -------
        str
            Description of what the rule validates
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert rule to dictionary representation.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the rule
        """
        return {
            "rule_name": self.rule_name,
            "rule_type": self.rule_type.value,
            "enabled": self.enabled,
            "rule_code": self.rule_code.name if self.rule_code else None,
            "description": self.get_description(),
        }


class RequiredRule(ValidationRule):
    """Rule to check if values are required (non-null, non-empty)."""

    def __init__(self, enabled: bool = True):
        super().__init__("Required", enabled, rule_code=RuleCode.REQUIRED)
        self.rule_type = RuleType.REQUIRED

    def validate(self, series: pd.Series) -> ValidationResult:
        """Check for missing or empty values."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        for idx, value in series.items():
            if pd.isna(value):
                error_indices.append(idx)
                error_messages.append("Value is null/NaN")
            elif pd.api.types.is_string_dtype(series) and str(value).strip() == "":
                error_indices.append(idx)
                error_messages.append("Value is empty string")

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        return "Values must be non-null and non-empty"


class UniqueRule(ValidationRule):
    """Rule to check for unique values."""

    def __init__(self, enabled: bool = True):
        super().__init__("Unique", enabled, rule_code=RuleCode.UNIQUE)
        self.rule_type = RuleType.UNIQUE

    def validate(self, series: pd.Series) -> ValidationResult:
        """Check for duplicate values."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        # Remove null values for uniqueness check
        clean_series = series.dropna()
        if clean_series.empty:
            return ValidationResult(True, 0, [], [], self.rule_name)

        # Find duplicates
        duplicated_mask = clean_series.duplicated(keep=False)
        error_indices = clean_series[duplicated_mask].index.tolist()

        # Create error messages for each duplicate
        error_messages = []
        value_counts = clean_series.value_counts()
        for idx in error_indices:
            value = clean_series.loc[idx]
            count = value_counts[value]
            error_messages.append(f"Duplicate value '{value}' (appears {count} times)")

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        return "All values must be unique"


class DatatypeRule(ValidationRule):
    """Rule to validate primitive data types (int, float, date, bool)."""

    def __init__(self, expected_type: str, enabled: bool = True):
        """
        Initialize datatype validation rule.

        Parameters
        ----------
        expected_type : str
            Expected primitive data type ('int', 'float', 'date', 'bool')
        enabled : bool, default True
            Whether the rule is enabled
        """
        # Map to specific RuleCode
        code_map = {
            "int": RuleCode.DATATYPE_INT,
            "float": RuleCode.DATATYPE_FLOAT,
            "date": RuleCode.DATATYPE_DATE,
            "bool": RuleCode.DATATYPE_BOOL,
        }
        super().__init__(
            f"Datatype ({expected_type})",
            enabled,
            rule_code=code_map.get(expected_type),
        )
        self.expected_type = expected_type
        self.rule_type = RuleType.DATATYPE

    def validate(self, series: pd.Series) -> ValidationResult:
        """Validate data type and format."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        # Normalize integer dtype if required
        series = DataHelper.normalize_int_dtype_vectorized(series, safe_mode=False)
        for idx, value in series.items():
            if pd.isna(value):
                continue  # Skip null values (handled by RequiredRule)

            if not self._is_valid_type(value):
                error_indices.append(idx)
                error_messages.append(
                    f"Value '{value}' is not valid {self.expected_type}"
                )

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def _is_valid_type(self, value: Any) -> bool:
        """Check if value matches expected type."""
        try:
            if self.expected_type == "int":
                return isinstance(value, (int, np.integer)) or str(value).isdigit()
            elif self.expected_type == "float":
                return isinstance(
                    value, (float, np.floating)
                ) or self._is_numeric_string(value)
            elif self.expected_type == "date":
                return self._is_valid_date(value)
            elif self.expected_type == "bool":
                return isinstance(value, (bool, np.bool_)) or str(value).lower() in [
                    "true",
                    "false",
                    "1",
                    "0",
                    "yes",
                    "no",
                ]
            else:
                return True  # Unknown type, assume valid
        except (ValueError, TypeError):
            return False

    def _is_numeric_string(self, value: str) -> bool:
        """Check if string represents a valid number."""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def _is_valid_date(self, value: Any) -> bool:
        """Check if value is a valid date."""
        try:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                return True
            pd.to_datetime(str(value))
            return True
        except (ValueError, TypeError, pd.errors.ParserError):
            return False

    def get_description(self) -> str:
        return f"Values must be valid primitive type: {self.expected_type}"


class FormatRule(ValidationRule):
    """Rule to validate data format for strings using extensible format validators."""

    # Registry of available format validators
    _validators = {
        "email": EmailValidator,
        "phone": PhoneValidator,
        "url": URLValidator,
        "ip": IPValidator,
        "credit_card": CreditCardValidator,
        "postal_code": PostalCodeValidator,
        "ssn": SSNValidator,
        "uuid": UUIDValidator,
    }

    def __init__(self, format_type: str, enabled: bool = True):
        """
        Initialize format validation rule.

        Parameters
        ----------
        format_type : str
            Type of format to validate ('email', 'phone', 'url', 'ip', 'credit_card', 'postal_code', 'ssn', 'uuid')
        enabled : bool, default True
            Whether the rule is enabled
        """
        # Map format type to RuleCode
        code_map = {
            "email": RuleCode.FORMAT_EMAIL,
            "phone": RuleCode.FORMAT_PHONE,
            "url": RuleCode.FORMAT_URL,
            "ip": RuleCode.FORMAT_IP,
            "credit_card": RuleCode.FORMAT_CREDIT_CARD,
            "postal_code": RuleCode.FORMAT_POSTAL_CODE,
            "ssn": RuleCode.FORMAT_SSN,
            "uuid": RuleCode.FORMAT_UUID,
        }

        if format_type not in self._validators:
            raise ValueError(
                f"Unsupported format type: {format_type}. Available types: {list(self._validators.keys())}"
            )

        super().__init__(
            f"Format ({format_type})", enabled, rule_code=code_map.get(format_type)
        )
        self.format_type = format_type
        self.rule_type = RuleType.FORMAT
        self.validator = self._validators[format_type]()

    def validate(self, series: pd.Series) -> ValidationResult:
        """Validate values against the specified format."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        for idx, value in series.items():
            if pd.isna(value):
                continue

            if not self.validator.validate(str(value)):
                error_indices.append(idx)
                error_messages.append(
                    f"Value '{value}' is not valid {self.format_type} format"
                )

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        return self.validator.get_description()

    @classmethod
    def get_supported_formats(cls) -> List[str]:
        """Get list of supported format types."""
        return list(cls._validators.keys())

    @classmethod
    def register_format_validator(cls, format_type: str, validator_class: type) -> None:
        """
        Register a new format validator.

        Parameters
        ----------
        format_type : str
            Name of the format type
        validator_class : type
            Class that implements FormatValidator interface
        """
        if not issubclass(validator_class, FormatValidator):
            raise ValueError("Validator class must inherit from FormatValidator")

        cls._validators[format_type] = validator_class
        logger.info(f"Registered new format validator: {format_type}")

    @classmethod
    def unregister_format_validator(cls, format_type: str) -> bool:
        """
        Unregister a format validator.

        Parameters
        ----------
        format_type : str
            Name of the format type to unregister

        Returns
        -------
        bool
            True if validator was removed, False if not found
        """
        if format_type in cls._validators:
            del cls._validators[format_type]
            logger.info(f"Unregistered format validator: {format_type}")
            return True
        return False


class MinMaxRule(ValidationRule):
    """Rule to validate numeric values within min/max bounds."""

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        enabled: bool = True,
    ):
        """
        Initialize min/max validation rule.

        Parameters
        ----------
        min_value : float, optional
            Minimum allowed value
        max_value : float, optional
            Maximum allowed value
        enabled : bool, default True
            Whether the rule is enabled
        """
        super().__init__("Min/Max Range", enabled, rule_code=RuleCode.MIN_MAX)
        self.min_value = min_value
        self.max_value = max_value
        self.rule_type = RuleType.MIN_MAX

    def validate(self, series: pd.Series) -> ValidationResult:
        """Validate values are within min/max bounds."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        for idx, value in series.items():
            if pd.isna(value):
                continue  # Skip null values

            try:
                numeric_value = float(value)

                if self.min_value is not None and numeric_value < self.min_value:
                    error_indices.append(idx)
                    error_messages.append(
                        f"Value {numeric_value} is below minimum {self.min_value}"
                    )
                elif self.max_value is not None and numeric_value > self.max_value:
                    error_indices.append(idx)
                    error_messages.append(
                        f"Value {numeric_value} is above maximum {self.max_value}"
                    )
            except (ValueError, TypeError):
                error_indices.append(idx)
                error_messages.append(f"Value '{value}' is not numeric")

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        bounds = []
        if self.min_value is not None:
            bounds.append(f"min: {self.min_value}")
        if self.max_value is not None:
            bounds.append(f"max: {self.max_value}")
        return f"Values must be within range ({', '.join(bounds)})"


class ValidValuesRule(ValidationRule):
    """Rule to validate values against a predefined list."""

    def __init__(self, valid_values: List[Any], enabled: bool = True):
        """
        Initialize valid values rule.

        Parameters
        ----------
        valid_values : List[Any]
            List of valid values
        enabled : bool, default True
            Whether the rule is enabled
        """
        super().__init__("Valid Values", enabled, rule_code=RuleCode.VALID_VALUES)
        self.valid_values = set(valid_values)  # Use set for O(1) lookup
        self.rule_type = RuleType.VALID_VALUES

    def validate(self, series: pd.Series) -> ValidationResult:
        """Validate values are in the allowed list."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        for idx, value in series.items():
            if pd.isna(value):
                continue  # Skip null values

            if value not in self.valid_values:
                error_indices.append(idx)
                error_messages.append(
                    f"Value '{value}' is not in allowed values: {sorted(self.valid_values)}"
                )

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        return f"Values must be one of: {sorted(self.valid_values)}"


class RegexRule(ValidationRule):
    """Rule to validate values against a regex pattern."""

    def __init__(self, pattern: str, enabled: bool = True):
        """
        Initialize regex validation rule.

        Parameters
        ----------
        pattern : str
            Regex pattern to match against
        enabled : bool, default True
            Whether the rule is enabled
        """
        super().__init__("Regex Pattern", enabled, rule_code=RuleCode.REGEX)
        self.pattern = pattern
        self.compiled_pattern = re.compile(pattern)
        self.rule_type = RuleType.REGEX

    def validate(self, series: pd.Series) -> ValidationResult:
        """Validate values match the regex pattern."""
        if not self.enabled:
            return ValidationResult(True, 0, [], [], self.rule_name)

        error_indices = []
        error_messages = []

        for idx, value in series.items():
            if pd.isna(value):
                continue  # Skip null values

            if not self.compiled_pattern.match(str(value)):
                error_indices.append(idx)
                error_messages.append(
                    f"Value '{value}' does not match pattern: {self.pattern}"
                )

        return ValidationResult(
            is_valid=len(error_indices) == 0,
            error_count=len(error_indices),
            error_indices=error_indices,
            error_messages=error_messages,
            rule_name=self.rule_name,
        )

    def get_description(self) -> str:
        return f"Values must match regex pattern: {self.pattern}"


class ValidationRuleRegistry:
    """Registry for managing validation rules."""

    def __init__(self):
        """Initialize the rule registry."""
        self._rules: Dict[str, ValidationRule] = {}
        self._rules_by_code: Dict[RuleCode, ValidationRule] = {}
        self._system_rules: Set[str] = set()

    def register_rule(self, rule: ValidationRule, is_system_rule: bool = False) -> None:
        """
        Register a validation rule.

        Parameters
        ----------
        rule : ValidationRule
            Rule to register
        is_system_rule : bool, default False
            Whether this is a system default rule
        """
        self._rules[rule.rule_name] = rule
        if getattr(rule, "rule_code", None) is not None:
            self._rules_by_code[rule.rule_code] = rule
        if is_system_rule:
            self._system_rules.add(rule.rule_name)

    def get_rule(self, rule_name: str) -> Optional[ValidationRule]:
        """
        Get a registered rule by name.

        Parameters
        ----------
        rule_name : str
            Name of the rule to retrieve

        Returns
        -------
        Optional[ValidationRule]
            The rule if found, None otherwise
        """
        return self._rules.get(rule_name)

    def get_rule_by_code(self, code: Union[str, RuleCode]) -> Optional[ValidationRule]:
        """Retrieve a rule by RuleCode (enum or name)."""
        if isinstance(code, RuleCode):
            return self._rules_by_code.get(code)
        try:
            rc = RuleCode[code]
        except Exception:
            return None
        return self._rules_by_code.get(rc)

    def get_all_rules(self) -> Dict[str, ValidationRule]:
        """
        Get all registered rules.

        Returns
        -------
        Dict[str, ValidationRule]
            Dictionary of all registered rules
        """
        return self._rules.copy()

    def get_system_rules(self) -> Dict[str, ValidationRule]:
        """
        Get all system default rules.

        Returns
        -------
        Dict[str, ValidationRule]
            Dictionary of system rules
        """
        return {
            name: rule
            for name, rule in self._rules.items()
            if name in self._system_rules
        }

    def get_custom_rules(self) -> Dict[str, ValidationRule]:
        """
        Get all custom (non-system) rules.

        Returns
        -------
        Dict[str, ValidationRule]
            Dictionary of custom rules
        """
        return {
            name: rule
            for name, rule in self._rules.items()
            if name not in self._system_rules
        }

    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove a rule from the registry.

        Parameters
        ----------
        rule_name : str
            Name of the rule to remove

        Returns
        -------
        bool
            True if rule was removed, False if not found
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            self._system_rules.discard(rule_name)
            return True
        return False


# Global registry instance
rule_registry = ValidationRuleRegistry()

# Register system default rules
rule_registry.register_rule(RequiredRule(), is_system_rule=True)
rule_registry.register_rule(UniqueRule(), is_system_rule=True)
rule_registry.register_rule(DatatypeRule("int"), is_system_rule=True)
rule_registry.register_rule(DatatypeRule("float"), is_system_rule=True)
rule_registry.register_rule(DatatypeRule("date"), is_system_rule=True)
rule_registry.register_rule(DatatypeRule("bool"), is_system_rule=True)

# Register format rules
rule_registry.register_rule(FormatRule("email"), is_system_rule=True)
rule_registry.register_rule(FormatRule("phone"), is_system_rule=True)
rule_registry.register_rule(FormatRule("url"), is_system_rule=True)
rule_registry.register_rule(FormatRule("ip"), is_system_rule=True)
rule_registry.register_rule(FormatRule("credit_card"), is_system_rule=True)
rule_registry.register_rule(FormatRule("postal_code"), is_system_rule=True)
rule_registry.register_rule(FormatRule("ssn"), is_system_rule=True)
rule_registry.register_rule(FormatRule("uuid"), is_system_rule=True)

# Optionally pre-register parameterized rule prototypes by alias for lookups
# Note: these are templates; actual parameterized instances are constructed
#       per-field using FieldDefinition.metadata in the calculator when needed.
rule_registry.register_rule(MinMaxRule(), is_system_rule=False)
rule_registry.register_rule(ValidValuesRule(valid_values=[]), is_system_rule=False)
rule_registry.register_rule(RegexRule(pattern=r""), is_system_rule=False)
