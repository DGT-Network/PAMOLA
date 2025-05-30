"""
Currency utilities for the project.

This module provides utility functions for working with currency data,
including currency detection, parsing, and analysis functions.
"""

import logging
import re
from decimal import Decimal
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)

# Common currency symbols and their ISO codes
CURRENCY_SYMBOLS = {
    '$': 'USD',
    '€': 'EUR',
    '£': 'GBP',
    '¥': 'JPY',
    '₽': 'RUB',
    '₹': 'INR',
    '₩': 'KRW',
    '₺': 'TRY',
    '₴': 'UAH',
    '฿': 'THB',
    'CHF': 'CHF',  # Special case where the code appears directly
    'zł': 'PLN',
    'kr': 'NOK',  # Ambiguous - could be NOK, SEK, DKK
    'R$': 'BRL',
    'C$': 'CAD',
    'A$': 'AUD',
    'HK$': 'HKD',
    '₿': 'BTC',  # Bitcoin
}

# Currency field name patterns
CURRENCY_FIELD_PATTERNS = [
    r'(?i).*amount.*',
    r'(?i).*price.*',
    r'(?i).*cost.*',
    r'(?i).*currency.*',
    r'(?i).*total.*',
    r'(?i).*salary.*',
    r'(?i).*income.*',
    r'(?i).*revenue.*',
    r'(?i).*expense.*',
    r'(?i).*budget.*',
    r'(?i).*pay.*',
    r'(?i).*cash.*',
    r'(?i).*money.*',
    r'(?i).*fund.*',
    r'(?i).*monetar.*',
    r'(?i).*dollar.*',
    r'(?i).*euro.*',
    r'(?i).*pound.*',
    r'(?i).*yen.*',
    r'(?i).*ruble.*',
    r'(?i).*usd.*',
    r'(?i).*eur.*',
    r'(?i).*gbp.*',
    r'(?i).*jpy.*',
    r'(?i).*rub.*',
]

# Locale formatting information
LOCALE_MAPPINGS = {
    'en_US': {'decimal': '.', 'thousands': ',', 'locale': 'en_US.UTF-8'},
    'fr_FR': {'decimal': ',', 'thousands': ' ', 'locale': 'fr_FR.UTF-8'},
    'de_DE': {'decimal': ',', 'thousands': '.', 'locale': 'de_DE.UTF-8'},
    'ru_RU': {'decimal': ',', 'thousands': ' ', 'locale': 'ru_RU.UTF-8'},
    'ja_JP': {'decimal': '.', 'thousands': ',', 'locale': 'ja_JP.UTF-8'},
    'en_GB': {'decimal': '.', 'thousands': ',', 'locale': 'en_GB.UTF-8'},
    'es_ES': {'decimal': ',', 'thousands': '.', 'locale': 'es_ES.UTF-8'},
    'it_IT': {'decimal': ',', 'thousands': '.', 'locale': 'it_IT.UTF-8'},
    'zh_CN': {'decimal': '.', 'thousands': ',', 'locale': 'zh_CN.UTF-8'},
}


def is_currency_field(field_name: str) -> bool:
    """
    Check if a field name matches common currency field patterns.

    Parameters:
    -----------
    field_name : str
        Name of the field to check

    Returns:
    --------
    bool
        True if the field name matches currency patterns, False otherwise
    """
    return any(re.match(pattern, field_name) for pattern in CURRENCY_FIELD_PATTERNS)


def extract_currency_symbol(value: str) -> Tuple[str, Optional[str]]:
    """
    Extract currency symbol from a string value.

    Parameters:
    -----------
    value : str
        String value to extract currency symbol from

    Returns:
    --------
    Tuple[str, Optional[str]]
        Tuple of (cleaned value, currency symbol or None)
    """
    # Handle None or non-string values
    if value is None or not isinstance(value, str):
        return str(value) if value is not None else "", None

    # Check for currency codes at the start or end
    for currency_symbol, iso_code in CURRENCY_SYMBOLS.items():
        # Check for symbol at start
        if value.startswith(currency_symbol):
            cleaned_value = value[len(currency_symbol):].strip()
            return cleaned_value, iso_code

        # Check for symbol at end
        if value.endswith(currency_symbol):
            cleaned_value = value[:-len(currency_symbol)].strip()
            return cleaned_value, iso_code

        # Check for symbol within the value with spaces
        symbol_pattern = f" {re.escape(currency_symbol)} "
        if symbol_pattern in value:
            cleaned_value = value.replace(symbol_pattern, "").strip()
            return cleaned_value, iso_code

    # Special handling for 3-letter currency codes like USD, EUR, etc.
    # This should be more conservative to avoid false positives
    for iso_code in set(CURRENCY_SYMBOLS.values()):
        # Check for ISO code at start with space or end with space
        if value.startswith(f"{iso_code} "):
            cleaned_value = value[len(iso_code) + 1:].strip()
            return cleaned_value, iso_code

        if value.endswith(f" {iso_code}"):
            cleaned_value = value[:-len(iso_code) - 1].strip()
            return cleaned_value, iso_code

    # No currency symbol found
    return value, None


def normalize_currency_value(value: Any, locale: str = 'en_US') -> Tuple[Optional[float], Optional[str], bool]:
    """
    Normalize a currency value to a float and extract its currency code.

    Parameters:
    -----------
    value : Any
        Value to normalize
    locale : str
        Locale to use for parsing (default: 'en_US')

    Returns:
    --------
    Tuple[Optional[float], Optional[str], bool]
        Tuple of (normalized value as float, currency code, is_valid flag)
    """
    # Handle None case
    if value is None:
        return None, None, False

    # Convert to string if not already
    if not isinstance(value, str):
        # If it's already a number, just return it
        if isinstance(value, (int, float, Decimal)):
            return float(value), None, True
        value = str(value)

    # Remove extra whitespace
    value = value.strip()
    if not value:
        return None, None, False

    # Extract currency symbol
    cleaned_value, currency_code = extract_currency_symbol(value)

    # Get locale decimal and thousands separators
    locale_info = LOCALE_MAPPINGS.get(locale, LOCALE_MAPPINGS['en_US'])
    decimal_sep = locale_info['decimal']
    thousands_sep = locale_info['thousands']

    # Remove thousands separators and normalize decimal separator
    cleaned_value = cleaned_value.replace(thousands_sep, '')
    if decimal_sep != '.':
        cleaned_value = cleaned_value.replace(decimal_sep, '.')

    # Remove any remaining non-numeric chars except decimal point
    # But keep negative sign if present
    is_negative = cleaned_value.startswith('-')
    cleaned_value = re.sub(r'[^\d.]', '', cleaned_value)

    # Reapply negative sign if needed
    if is_negative:
        cleaned_value = '-' + cleaned_value

    # Try to convert to float
    try:
        numeric_value = float(cleaned_value)
        return numeric_value, currency_code, True
    except (ValueError, TypeError):
        logger.debug(f"Failed to parse currency value: {value}")
        return None, currency_code, False


def parse_currency_field(df: pd.DataFrame, field_name: str, locale: str = 'en_US') -> Tuple[pd.Series, Dict[str, int]]:
    """
    Parse a currency field and extract normalized values and currency codes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the currency field
    field_name : str
        Name of the currency field
    locale : str
        Locale to use for parsing (default: 'en_US')

    Returns:
    --------
    Tuple[pd.Series, Dict[str, int]]
        Tuple of (normalized values as Series, currency_counts dict)
    """
    if field_name not in df.columns:
        raise ValueError(f"Field {field_name} not found in DataFrame")

    # Initialize results
    values = []
    currencies = []
    currency_counts = {}
    valid_flags = []

    # Process each value
    for val in df[field_name]:
        numeric_val, currency_code, is_valid = normalize_currency_value(val, locale)
        values.append(numeric_val)
        currencies.append(currency_code)
        valid_flags.append(is_valid)

        # Count currency codes
        if currency_code:
            currency_counts[currency_code] = currency_counts.get(currency_code, 0) + 1

    # Create a Series with the normalized values
    normalized_values = pd.Series(values, index=df.index)

    # Store currency codes and valid flags as attributes
    normalized_values.currencies = currencies
    normalized_values.valid_flags = valid_flags

    return normalized_values, currency_counts


def analyze_currency_stats(values: Union[pd.Series, np.ndarray],
                           currency_counts: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Analyze currency values and compute statistics.

    Parameters:
    -----------
    values : Union[pd.Series, np.ndarray]
        Normalized currency values
    currency_counts : Dict[str, int], optional
        Counts of detected currencies

    Returns:
    --------
    Dict[str, Any]
        Dictionary of currency statistics
    """
    # Initialize result dictionary with explicit Dict type for IDE clarity
    result_stats: Dict[str, Any] = {}

    # Handle empty or all-null case
    valid_values = pd.Series(values).dropna()
    if len(valid_values) == 0:
        result_stats["min"] = None
        result_stats["max"] = None
        result_stats["mean"] = None
        result_stats["median"] = None
        result_stats["std"] = None
        result_stats["skewness"] = None
        result_stats["kurtosis"] = None
        result_stats["valid_count"] = 0
        result_stats["zero_count"] = 0
        result_stats["zero_percentage"] = 0.0
        result_stats["negative_count"] = 0
        result_stats["negative_percentage"] = 0.0
        result_stats["currency_distribution"] = currency_counts or {}
        result_stats["multi_currency"] = bool(currency_counts and len(currency_counts) > 1)
        return result_stats

    # Calculate basic statistics
    result_stats["min"] = float(valid_values.min())
    result_stats["max"] = float(valid_values.max())
    result_stats["mean"] = float(valid_values.mean())
    result_stats["median"] = float(valid_values.median())
    result_stats["std"] = float(valid_values.std()) if len(valid_values) > 1 else 0.0
    result_stats["valid_count"] = len(valid_values)

    # Count zeros and negatives using pure Python methods to avoid IDE type issues
    zero_count = 0
    negative_count = 0

    # Iterate manually to avoid pandas boolean operations that confuse the IDE
    for val in valid_values:
        if val == 0:
            zero_count += 1
        if val < 0:
            negative_count += 1

    result_stats["zero_count"] = zero_count
    result_stats["negative_count"] = negative_count

    # Calculate derived statistics
    valid_count = len(valid_values)
    result_stats["zero_percentage"] = (zero_count / valid_count) * 100 if valid_count > 0 else 0.0
    result_stats["negative_percentage"] = (negative_count / valid_count) * 100 if valid_count > 0 else 0.0

    # Calculate distribution statistics if we have enough data
    if len(valid_values) >= 3:
        result_stats["skewness"] = float(valid_values.skew())
        result_stats["kurtosis"] = float(valid_values.kurtosis())
    else:
        result_stats["skewness"] = 0.0
        result_stats["kurtosis"] = 0.0

    # Add currency distribution information
    result_stats["currency_distribution"] = currency_counts or {}
    result_stats["multi_currency"] = bool(currency_counts and len(currency_counts) > 1)

    return result_stats


def detect_currency_from_sample(df: pd.DataFrame, field_name: str, sample_size: int = 100) -> str:
    """
    Detect the most likely currency from a sample of data.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the currency field
    field_name : str
        Name of the currency field
    sample_size : int
        Number of samples to check (default: 100)

    Returns:
    --------
    str
        Detected currency code, or 'UNKNOWN' if none detected
    """
    if field_name not in df.columns:
        return 'UNKNOWN'

    # Take a sample of the data
    sample = df[field_name].dropna().head(sample_size)

    # Count currency codes
    currency_counts = {}
    for val in sample:
        if not isinstance(val, str):
            continue

        _, currency_code = extract_currency_symbol(str(val))
        if currency_code:
            currency_counts[currency_code] = currency_counts.get(currency_code, 0) + 1

    # Return the most common currency code, or UNKNOWN if none detected
    if not currency_counts:
        return 'UNKNOWN'

    return max(currency_counts.items(), key=lambda x: x[1])[0]


def generate_currency_samples(stats: Dict[str, Any], count: int = 10) -> Dict[str, List[float]]:
    """
    Generate sample currency values based on statistics for dictionary output.

    Parameters:
    -----------
    stats : Dict[str, Any]
        Dictionary of currency statistics
    count : int
        Number of samples to generate (default: 10)

    Returns:
    --------
    Dict[str, List[float]]
        Dictionary of currency samples
    """
    samples = {}

    # Check if we have valid statistics
    if stats.get('valid_count', 0) == 0:
        return samples

    # Generate samples based on min, max, mean, and std
    min_val = stats.get('min', 0)
    max_val = stats.get('max', 1)
    mean = stats.get('mean', (min_val + max_val) / 2)
    std = stats.get('std', (max_val - min_val) / 4)

    # Ensure std is not zero to avoid sampling issues
    if std <= 0:
        std = (max_val - min_val) / 4
        if std <= 0:
            std = 1.0

    # Generate samples from normal distribution, clipped to min/max range
    try:
        normal_samples = np.random.normal(mean, std, count)
        clipped_samples = np.clip(normal_samples, min_val, max_val)
        samples['normal'] = clipped_samples.tolist()
    except Exception as e:
        logger.warning(f"Error generating normal samples: {e}")
        samples['normal'] = [mean] * count

    # Add some boundary samples
    samples['boundary'] = [min_val, max_val]

    # Add some special samples
    if stats.get('zero_count', 0) > 0:
        samples['special'] = [0.0]

    if stats.get('negative_count', 0) > 0:
        samples['special'] = samples.get('special', []) + [-abs(mean / 2)]

    return samples


def create_empty_currency_stats() -> Dict[str, Any]:
    """
    Create an empty dictionary of currency statistics.

    Returns:
    --------
    Dict[str, Any]
        Empty dictionary of currency statistics
    """
    return {
        'min': None,
        'max': None,
        'mean': None,
        'median': None,
        'std': None,
        'skewness': None,
        'kurtosis': None,
        'valid_count': 0,
        'zero_count': 0,
        'zero_percentage': 0.0,
        'negative_count': 0,
        'negative_percentage': 0.0,
        'currency_distribution': {},
        'multi_currency': False,
        'outliers': {
            'iqr': None,
            'lower_bound': None,
            'upper_bound': None,
            'count': 0,
            'percentage': 0.0
        },
        'normality': {
            'is_normal': False,
            'message': 'Insufficient data for normality testing'
        }
    }