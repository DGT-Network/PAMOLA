"""
PAMOLA.CORE - Privacy-Aware Management of Large Anonymization
------------------------------------------------------------
Module:        Type-Specific Validators
Package:       pamola_core.anonymization.commons.validation
Version:       1.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  Validators for specialized data types including network identifiers,
  geographic data, temporal sequences, and financial data. Provides
  comprehensive validation for domain-specific data formats with
  detailed error reporting.

Key Features:
  - Network identifier validation (IPv4, IPv6, MAC, URL, email)
  - Geographic data validation (coordinates, addresses, postal codes)
  - Temporal sequence validation with range checking
  - Financial data validation (amounts, accounts, transaction IDs)
  - Pattern-based validation with regex support
  - Sample-based validation for performance
  - Detailed validation results with warnings

Design Principles:
  - Performance: Sample-based validation for large datasets
  - Extensibility: Easy to add new specialized types
  - Clarity: Clear error messages with examples
  - Consistency: Standardized validation result format

Usage:
  Used by anonymization operations to validate specialized data
  types before processing. Ensures data conforms to expected
  formats and ranges.

Dependencies:
  - pandas - Series operations
  - numpy - Numeric operations
  - re - Pattern matching
  - logging - Error reporting
  - ipaddress - IP address validation
  - typing - Type hints
  - datetime - Date validation

Changelog:
  1.0.0 - Initial implementation extracted from validation_utils
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd

# Import base classes and decorators
from .base import ValidationResult, BaseValidator
from .decorators import validation_handler, standard_validator

# Optional imports for specialized validation
try:
    import ipaddress

    IPADDRESS_AVAILABLE = True
except ImportError:
    IPADDRESS_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

# Network patterns
NETWORK_PATTERNS = {
    'ipv4': re.compile(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    ),
    'mac': re.compile(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$'),
    'url': re.compile(r'^https?://[^\s/$.?#].\S*$', re.IGNORECASE),
    'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
}

# Geographic patterns
GEO_PATTERNS = {
    'postal_code': re.compile(r'^[A-Za-z0-9\s\-]{3,10}$'),
    'coordinates': re.compile(r'^-?\d+\.?\d*,-?\d+\.?\d*$')
}

# Financial patterns
FINANCIAL_PATTERNS = {
    'credit_card': re.compile(r'^\d{13,19}$'),
    'iban': re.compile(r'^[A-Z]{2}\d{2}[A-Z0-9]+$'),
    'swift': re.compile(r'^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$')
}

# Currency codes (ISO 4217 common subset)
COMMON_CURRENCIES = {
    'USD', 'EUR', 'GBP', 'JPY', 'CAD', 'AUD', 'CHF', 'CNY',
    'SEK', 'NZD', 'MXN', 'SGD', 'HKD', 'NOK', 'KRW', 'TRY',
    'RUB', 'INR', 'BRL', 'ZAR'
}

# Reasonable date ranges
MIN_REASONABLE_DATE = datetime(1900, 1, 1)
MAX_REASONABLE_DATE = datetime(2100, 1, 1)


# =============================================================================
# Network Validators
# =============================================================================

class NetworkValidator(BaseValidator):
    """Validator for network identifiers."""

    def __init__(self, network_type: str = 'ipv4',
                 sample_size: int = 100,
                 strict: bool = False):
        """
        Initialize network validator.

        Args:
            network_type: Type of network data (ipv4, ipv6, mac, url, email)
            sample_size: Number of records to sample for validation
            strict: Whether to validate all records or just sample
        """
        super().__init__()
        self.network_type = network_type
        self.sample_size = sample_size
        self.strict = strict

        if network_type not in ['ipv4', 'ipv6', 'mac', 'url', 'email']:
            raise ValueError(f"Invalid network_type: {network_type}")

    @validation_handler()
    def validate(self, data: pd.Series, **kwargs) -> ValidationResult:
        """
        Validate network identifier format.

        Args:
            data: Series containing network identifiers
            **kwargs: Additional validation parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        result.field_name = kwargs.get('field_name', 'network_field')

        # Sample data for performance unless strict mode
        if self.strict:
            sample_data = data.dropna()
        else:
            non_null = data.dropna()
            if len(non_null) > self.sample_size:
                sample_data = non_null.sample(
                    n=self.sample_size,
                    random_state=42
                )
            else:
                sample_data = non_null

        # Track validation stats
        invalid_count = 0
        invalid_examples = []

        # Validate based on network type
        if self.network_type == 'ipv4':
            pattern = NETWORK_PATTERNS['ipv4']
            for idx, value in sample_data.items():
                if not isinstance(value, str) or not pattern.match(value):
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(value)

        elif self.network_type == 'ipv6':
            if not IPADDRESS_AVAILABLE:
                result.add_error("ipaddress module not available for IPv6 validation")
                return result

            for idx, value in sample_data.items():
                try:
                    ipaddress.IPv6Address(str(value))
                except (ValueError, ipaddress.AddressValueError):
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(value)

        elif self.network_type in ['mac', 'url', 'email']:
            pattern = NETWORK_PATTERNS[self.network_type]
            for idx, value in sample_data.items():
                if not isinstance(value, str) or not pattern.match(value):
                    invalid_count += 1
                    if len(invalid_examples) < 5:
                        invalid_examples.append(value)

        # Calculate validation rate
        total_checked = len(sample_data)
        if total_checked > 0:
            invalid_rate = invalid_count / total_checked
            result.details['validation_rate'] = 1 - invalid_rate
            result.details['invalid_count'] = invalid_count
            result.details['total_checked'] = total_checked

            if invalid_count > 0:
                result.details['invalid_examples'] = invalid_examples

                if invalid_rate > 0.1:  # More than 10% invalid
                    result.add_error(
                        f"High invalid rate for {self.network_type}: "
                        f"{invalid_rate:.1%}"
                    )
                else:
                    result.add_warning(
                        f"Found {invalid_count} invalid {self.network_type} values"
                    )

        return result


@standard_validator()
def validate_network_identifiers(data: pd.Series,
                                 network_type: str,
                                 strict: bool = False) -> ValidationResult:
    """
    Validate network identifier format.

    Args:
        data: Network identifier data
        network_type: Type of network identifier
        strict: Whether to validate all records

    Returns:
        ValidationResult with validation outcome
    """
    validator = NetworkValidator(network_type=network_type, strict=strict)
    return validator.validate(data)


# =============================================================================
# Geographic Validators
# =============================================================================

class GeographicValidator(BaseValidator):
    """Validator for geographic data."""

    def __init__(self, geo_type: str = 'coordinates',
                 sample_size: int = 100):
        """
        Initialize geographic validator.

        Args:
            geo_type: Type of geographic data
            sample_size: Number of records to sample
        """
        super().__init__()
        self.geo_type = geo_type
        self.sample_size = sample_size

        valid_types = ['coordinates', 'address', 'postal_code',
                       'latitude', 'longitude']
        if geo_type not in valid_types:
            raise ValueError(
                f"Invalid geo_type: {geo_type}. "
                f"Must be one of {valid_types}"
            )

    @validation_handler()
    def validate(self, data: pd.Series, **kwargs) -> ValidationResult:
        """
        Validate geographic data format.

        Args:
            data: Geographic data to validate
            **kwargs: Additional parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        result.field_name = kwargs.get('field_name', 'geo_field')

        # Sample data
        non_null = data.dropna()
        if len(non_null) == 0:
            result.add_warning("No non-null geographic data to validate")
            return result

        sample_data = non_null.sample(
            n=min(self.sample_size, len(non_null)),
            random_state=42
        )

        # Validate based on type
        if self.geo_type == 'coordinates':
            self._validate_coordinates(sample_data, result)
        elif self.geo_type in ['latitude', 'longitude']:
            self._validate_lat_lng(sample_data, result)
        elif self.geo_type == 'postal_code':
            self._validate_postal_code(sample_data, result)
        elif self.geo_type == 'address':
            self._validate_address(sample_data, result)

        return result

    def _validate_coordinates(self, data: pd.Series,
                              result: ValidationResult) -> None:
        """Validate coordinate pairs."""
        invalid_count = 0
        out_of_range = 0

        for idx, value in data.items():
            if isinstance(value, str):
                # Try to parse "lat,lng" format
                parts = value.replace(' ', '').split(',')
                if len(parts) != 2:
                    invalid_count += 1
                    continue

                try:
                    lat, lng = float(parts[0]), float(parts[1])
                    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                        out_of_range += 1
                except ValueError:
                    invalid_count += 1

            elif isinstance(value, (tuple, list)) and len(value) == 2:
                try:
                    lat, lng = float(value[0]), float(value[1])
                    if not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                        out_of_range += 1
                except (ValueError, TypeError):
                    invalid_count += 1
            else:
                invalid_count += 1

        total = len(data)
        result.details['invalid_format'] = invalid_count
        result.details['out_of_range'] = out_of_range
        result.details['total_checked'] = total

        if invalid_count > 0:
            error_rate = invalid_count / total
            if error_rate > 0.1:
                result.add_error(
                    f"High invalid coordinate rate: {error_rate:.1%}"
                )
            else:
                result.add_warning(
                    f"Found {invalid_count} invalid coordinates"
                )

        if out_of_range > 0:
            result.add_warning(
                f"Found {out_of_range} coordinates out of valid range"
            )

    def _validate_lat_lng(self, data: pd.Series,
                          result: ValidationResult) -> None:
        """Validate latitude or longitude values."""
        invalid_count = 0
        out_of_range = 0

        if self.geo_type == 'latitude':
            min_val, max_val = -90, 90
        else:  # longitude
            min_val, max_val = -180, 180

        for idx, value in data.items():
            try:
                num_val = float(value)
                if not (min_val <= num_val <= max_val):
                    out_of_range += 1
            except (ValueError, TypeError):
                invalid_count += 1

        total = len(data)
        if invalid_count > 0:
            error_rate = invalid_count / total
            if error_rate > 0.1:
                result.add_error(
                    f"High invalid {self.geo_type} rate: {error_rate:.1%}"
                )

        if out_of_range > 0:
            result.add_warning(
                f"Found {out_of_range} {self.geo_type} values out of range "
                f"[{min_val}, {max_val}]"
            )

    def _validate_postal_code(self, data: pd.Series,
                              result: ValidationResult) -> None:
        """Validate postal codes."""
        pattern = GEO_PATTERNS['postal_code']
        invalid_count = 0

        for idx, value in data.items():
            if not isinstance(value, str) or not pattern.match(value):
                invalid_count += 1

        if invalid_count > 0:
            result.add_warning(
                f"Found {invalid_count} invalid postal codes. "
                f"Note: validation uses generic pattern"
            )

    def _validate_address(self, data: pd.Series,
                          result: ValidationResult) -> None:
        """Basic address validation."""
        too_short = 0

        for idx, value in data.items():
            if not isinstance(value, str) or len(value.strip()) < 5:
                too_short += 1

        if too_short > 0:
            result.add_warning(
                f"Found {too_short} addresses shorter than 5 characters"
            )


@standard_validator()
def validate_geographic_data(data: pd.Series,
                             geo_type: str) -> ValidationResult:
    """
    Validate geographic data format.

    Args:
        data: Geographic data to validate
        geo_type: Type of geographic data

    Returns:
        ValidationResult with validation outcome
    """
    validator = GeographicValidator(geo_type=geo_type)
    return validator.validate(data)


# =============================================================================
# Temporal Validators
# =============================================================================

class TemporalValidator(BaseValidator):
    """Validator for temporal/datetime data."""

    def __init__(self, min_date: Optional[datetime] = None,
                 max_date: Optional[datetime] = None,
                 check_sequence: bool = False):
        """
        Initialize temporal validator.

        Args:
            min_date: Minimum allowed date
            max_date: Maximum allowed date
            check_sequence: Whether to check temporal ordering
        """
        super().__init__()
        self.min_date = min_date or MIN_REASONABLE_DATE
        self.max_date = max_date or MAX_REASONABLE_DATE
        self.check_sequence = check_sequence

    @validation_handler()
    def validate(self, data: pd.Series, **kwargs) -> ValidationResult:
        """
        Validate temporal data.

        Args:
            data: Temporal data to validate
            **kwargs: Additional parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        result.field_name = kwargs.get('field_name', 'temporal_field')

        # Try to convert to datetime
        try:
            datetime_data = pd.to_datetime(data, errors='coerce')
            null_count = datetime_data.isnull().sum()

            result.details['conversion_failures'] = int(null_count)
            result.details['conversion_rate'] = 1 - (null_count / len(data))

            if null_count > len(data) * 0.1:
                result.add_warning(
                    f"High datetime conversion failure rate: "
                    f"{null_count}/{len(data)}"
                )

            # Check date ranges
            valid_dates = datetime_data.dropna()
            if len(valid_dates) > 0:
                min_found = valid_dates.min()
                max_found = valid_dates.max()

                result.details['date_range'] = {
                    'min': min_found.isoformat(),
                    'max': max_found.isoformat()
                }

                if min_found < self.min_date:
                    result.add_error(
                        f"Dates before minimum allowed: "
                        f"{min_found} < {self.min_date}"
                    )

                if max_found > self.max_date:
                    result.add_error(
                        f"Dates after maximum allowed: "
                        f"{max_found} > {self.max_date}"
                    )

                # Check sequence if requested
                if self.check_sequence:
                    # Check if dates are sorted
                    is_sorted = valid_dates.is_monotonic_increasing
                    result.details['is_sorted'] = bool(is_sorted)

                    # Check for gaps
                    if len(valid_dates) > 1:
                        time_diffs = valid_dates.diff().dropna()
                        result.details['avg_time_diff'] = str(time_diffs.mean())
                        result.details['max_gap'] = str(time_diffs.max())

        except Exception as e:
            result.add_error(f"Failed to validate temporal data: {str(e)}")

        return result


@standard_validator()
def validate_temporal_sequence(data: pd.Series,
                               check_sequence: bool = False) -> ValidationResult:
    """
    Validate temporal sequence data.

    Args:
        data: Temporal data to validate
        check_sequence: Whether to check ordering

    Returns:
        ValidationResult with validation outcome
    """
    validator = TemporalValidator(check_sequence=check_sequence)
    return validator.validate(data)


# =============================================================================
# Financial Validators
# =============================================================================

class FinancialValidator(BaseValidator):
    """Validator for financial data."""

    def __init__(self, financial_type: str = 'amount',
                 allow_negative: bool = False,
                 currency: Optional[str] = None):
        """
        Initialize financial validator.

        Args:
            financial_type: Type of financial data
            allow_negative: Whether negative values are allowed
            currency: Expected currency code
        """
        super().__init__()
        self.financial_type = financial_type
        self.allow_negative = allow_negative
        self.currency = currency

        valid_types = ['amount', 'account', 'transaction_id',
                       'currency', 'credit_card']
        if financial_type not in valid_types:
            raise ValueError(
                f"Invalid financial_type: {financial_type}. "
                f"Must be one of {valid_types}"
            )

    @validation_handler()
    def validate(self, data: pd.Series, **kwargs) -> ValidationResult:
        """
        Validate financial data.

        Args:
            data: Financial data to validate
            **kwargs: Additional parameters

        Returns:
            ValidationResult with validation outcome
        """
        result = ValidationResult(is_valid=True)
        result.field_name = kwargs.get('field_name', 'financial_field')

        # Route to specific validator
        if self.financial_type == 'amount':
            self._validate_amount(data, result)
        elif self.financial_type == 'account':
            self._validate_account(data, result)
        elif self.financial_type == 'transaction_id':
            self._validate_transaction_id(data, result)
        elif self.financial_type == 'currency':
            self._validate_currency(data, result)
        elif self.financial_type == 'credit_card':
            self._validate_credit_card(data, result)

        return result

    def _validate_amount(self, data: pd.Series,
                         result: ValidationResult) -> None:
        """Validate monetary amounts."""
        invalid_count = 0
        negative_count = 0

        for idx, value in data.dropna().items():
            try:
                # Remove common currency symbols and separators
                clean_value = str(value).replace(',', '').replace('$', '')
                clean_value = clean_value.replace('€', '').replace('£', '')

                amount = float(clean_value)

                if amount < 0:
                    negative_count += 1
                    if not self.allow_negative:
                        invalid_count += 1

            except (ValueError, TypeError):
                invalid_count += 1

        total = len(data.dropna())
        if total > 0:
            result.details['negative_count'] = negative_count
            result.details['invalid_count'] = invalid_count

            if invalid_count > 0:
                error_rate = invalid_count / total
                if error_rate > 0.05:
                    result.add_error(
                        f"High invalid amount rate: {error_rate:.1%}"
                    )

            if negative_count > 0 and not self.allow_negative:
                result.add_error(
                    f"Found {negative_count} negative amounts"
                )

    def _validate_account(self, data: pd.Series,
                          result: ValidationResult) -> None:
        """Validate account numbers."""
        invalid_count = 0

        for idx, value in data.dropna().items():
            if not isinstance(value, str):
                invalid_count += 1
                continue

            # Basic validation: alphanumeric with possible separators
            clean_value = value.replace('-', '').replace(' ', '')
            if not clean_value.isalnum():
                invalid_count += 1

        if invalid_count > 0:
            result.add_warning(
                f"Found {invalid_count} invalid account numbers"
            )

    def _validate_transaction_id(self, data: pd.Series,
                                 result: ValidationResult) -> None:
        """Validate transaction IDs."""
        too_short = 0

        for idx, value in data.dropna().items():
            if not isinstance(value, str) or len(value) < 5:
                too_short += 1

        if too_short > 0:
            result.add_warning(
                f"Found {too_short} transaction IDs shorter than 5 characters"
            )

    def _validate_currency(self, data: pd.Series,
                           result: ValidationResult) -> None:
        """Validate currency codes."""
        invalid_count = 0
        unknown_currencies = set()

        for idx, value in data.dropna().items():
            if not isinstance(value, str) or len(value) != 3:
                invalid_count += 1
                continue

            value_upper = value.upper()
            if not value_upper.isalpha():
                invalid_count += 1
            elif value_upper not in COMMON_CURRENCIES:
                unknown_currencies.add(value_upper)

        if invalid_count > 0:
            result.add_warning(
                f"Found {invalid_count} invalid currency codes"
            )

        if unknown_currencies:
            result.add_warning(
                f"Found uncommon currency codes: "
                f"{', '.join(sorted(unknown_currencies)[:5])}"
            )

        if self.currency:
            # Check if all match expected currency
            expected_upper = self.currency.upper()
            non_matching = sum(
                1 for _, v in data.dropna().items()
                if isinstance(v, str) and v.upper() != expected_upper
            )
            if non_matching > 0:
                result.add_warning(
                    f"Found {non_matching} values not matching "
                    f"expected currency {expected_upper}"
                )

    def _validate_credit_card(self, data: pd.Series,
                              result: ValidationResult) -> None:
        """Basic credit card validation."""
        invalid_length = 0

        for idx, value in data.dropna().items():
            if isinstance(value, str):
                # Remove non-digits
                digits_only = re.sub(r'\D', '', value)
                if not (13 <= len(digits_only) <= 19):
                    invalid_length += 1
            else:
                invalid_length += 1

        if invalid_length > 0:
            result.add_warning(
                f"Found {invalid_length} credit card numbers with "
                f"invalid length (expected 13-19 digits)"
            )


@standard_validator()
def validate_financial_data(data: pd.Series,
                            financial_type: str,
                            allow_negative: bool = False) -> ValidationResult:
    """
    Validate financial data format.

    Args:
        data: Financial data to validate
        financial_type: Type of financial data
        allow_negative: Whether negative amounts are allowed

    Returns:
        ValidationResult with validation outcome
    """
    validator = FinancialValidator(
        financial_type=financial_type,
        allow_negative=allow_negative
    )
    return validator.validate(data)


# =============================================================================
# Composite Type Validator
# =============================================================================

class SpecializedTypeValidator(BaseValidator):
    """Unified validator for all specialized types."""

    def __init__(self, data_type: str,
                 validation_params: Optional[Dict[str, Any]] = None):
        """
        Initialize specialized type validator.

        Args:
            data_type: Type of specialized data
            validation_params: Additional validation parameters
        """
        super().__init__()
        self.data_type = data_type
        self.validation_params = validation_params or {}

        # Map data types to validators
        self.validators = {
            'network': NetworkValidator,
            'ip': NetworkValidator,  # Alias
            'geo': GeographicValidator,
            'geographic': GeographicValidator,  # Alias
            'temporal': TemporalValidator,
            'datetime': TemporalValidator,  # Alias
            'financial': FinancialValidator,
            'finance': FinancialValidator  # Alias
        }

    @validation_handler()
    def validate(self, data: pd.Series, **kwargs) -> ValidationResult:
        """
        Validate specialized data type.

        Args:
            data: Data to validate
            **kwargs: Additional parameters

        Returns:
            ValidationResult with validation outcome
        """
        # Find appropriate validator
        validator_class = self.validators.get(self.data_type)

        if not validator_class:
            raise ValueError(f"Unknown specialized data type: {self.data_type}")

        # Extract subtype parameters
        if self.data_type in ['network', 'ip']:
            subtype = self.validation_params.get(
                'network_type',
                self.validation_params.get('ip_type', 'ipv4')
            )
            validator = validator_class(network_type=subtype)

        elif self.data_type in ['geo', 'geographic']:
            subtype = self.validation_params.get('geo_type', 'coordinates')
            validator = validator_class(geo_type=subtype)

        elif self.data_type in ['temporal', 'datetime']:
            validator = validator_class(
                check_sequence=self.validation_params.get('check_sequence', False)
            )

        elif self.data_type in ['financial', 'finance']:
            subtype = self.validation_params.get('financial_type', 'amount')
            validator = validator_class(
                financial_type=subtype,
                allow_negative=self.validation_params.get('allow_negative', False)
            )
        else:
            validator = validator_class(**self.validation_params)

        # Run validation
        return validator.validate(data, **kwargs)


@standard_validator()
def validate_specialized_type(data: pd.Series,
                              data_type: str,
                              validation_params: Optional[Dict[str, Any]] = None,
                              field_name: Optional[str] = None) -> ValidationResult:
    """
    Validate specialized data types.

    Args:
        data: Data to validate
        data_type: Type of specialized data
        validation_params: Additional validation parameters
        field_name: Field name for reporting

    Returns:
        ValidationResult with validation outcome
    """
    validator = SpecializedTypeValidator(
        data_type=data_type,
        validation_params=validation_params
    )
    return validator.validate(data, field_name=field_name)


# Module exports
__all__ = [
    # Network validators
    'NetworkValidator',
    'validate_network_identifiers',

    # Geographic validators
    'GeographicValidator',
    'validate_geographic_data',

    # Temporal validators
    'TemporalValidator',
    'validate_temporal_sequence',

    # Financial validators
    'FinancialValidator',
    'validate_financial_data',

    # Composite validator
    'SpecializedTypeValidator',
    'validate_specialized_type',

    # Constants
    'NETWORK_PATTERNS',
    'GEO_PATTERNS',
    'FINANCIAL_PATTERNS',
    'COMMON_CURRENCIES'
]