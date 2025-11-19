"""
Phone number analysis utilities for the anonymization project.

This module provides analytical functions for phone number data fields,
including phone validation, component extraction, and messenger detection.
Functions in this module focus purely on analytics, without dependencies
on operation infrastructure, IO, or visualization components.

Main functions:
- is_valid_phone: Validate phone number format
- parse_phone_number: Parse phone into components
- identify_country_code: Extract country code
- identify_operator_code: Extract operator code
- detect_messenger_references: Identify messenger mentions in comments
- analyze_phone_field: Perform complete phone field analysis
- create_country_code_dictionary: Create frequency dictionary for country codes
- create_operator_code_dictionary: Create frequency dictionary for operator codes
- create_messenger_dictionary: Create frequency dictionary for messenger mentions
- normalize_phone: Normalize phone to E.164 format
- estimate_resources: Estimate resources for analysis
"""

import csv
import logging
import os
import re
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import phonenumbers

from pamola_core.common.regex.patterns import PhonePatterns

# Configure logger
logger = logging.getLogger(__name__)

# Default patterns for messenger detection
DEFAULT_MESSENGER_PATTERNS = {
    'telegram': [
        r'telegram', r'телеграм', r't\.me', r'tg', r'тг', r'@[\w\d_]{5,}',
        r'https?://t\.me', r'teleg'
    ],
    'whatsapp': [
        r'whatsapp', r'вотсап', r'wa\b', r'whats app', r'ватсап', r'whats[\s_-]?app',
        r'\bwa\b', r'\bwhap\b', r'вацап'
    ],
    'viber': [
        r'viber', r'вайбер', r'\bvb\b'
    ],
    'signal': [
        r'signal', r'сигнал'
    ],
    'wechat': [
        r'wechat', r'вичат'
    ],
    'line': [
        r'\bline\b', r'лайн месс'
    ],
    'other': [
        r'мессенджер', r'messenger', r'месс[еэ]нджер', r'соц[.]? сет', r'соцсет',
        r'discord', r'skype', r'скайп', r'facetime', r'facebook', r'фейсбук'
    ]
}

# Country code prefix mapping (most common country codes)
COUNTRY_PREFIX_MAP = {
        '1': 'US',    # USA/Canada (NANP)
        '7': 'RU',    # Russia
        '20': 'EG',   # Egypt
        '27': 'ZA',   # South Africa
        '30': 'GR',   # Greece
        '31': 'NL',   # Netherlands
        '32': 'BE',   # Belgium
        '33': 'FR',   # France
        '34': 'ES',   # Spain
        '36': 'HU',   # Hungary
        '39': 'IT',   # Italy
        '40': 'RO',   # Romania
        '41': 'CH',   # Switzerland
        '43': 'AT',   # Austria
        '44': 'GB',   # United Kingdom
        '45': 'DK',   # Denmark
        '46': 'SE',   # Sweden
        '47': 'NO',   # Norway
        '48': 'PL',   # Poland
        '49': 'DE',   # Germany
        '51': 'PE',   # Peru
        '52': 'MX',   # Mexico
        '54': 'AR',   # Argentina
        '55': 'BR',   # Brazil
        '56': 'CL',   # Chile
        '57': 'CO',   # Colombia
        '58': 'VE',   # Venezuela
        '60': 'MY',   # Malaysia
        '61': 'AU',   # Australia
        '62': 'ID',   # Indonesia
        '63': 'PH',   # Philippines
        '64': 'NZ',   # New Zealand
        '65': 'SG',   # Singapore
        '66': 'TH',   # Thailand
        '81': 'JP',   # Japan
        '82': 'KR',   # South Korea
        '84': 'VN',   # Vietnam
        '86': 'CN',   # China
        '90': 'TR',   # Turkey
        '91': 'IN',   # India
        '92': 'PK',   # Pakistan
        '93': 'AF',   # Afghanistan
        '94': 'LK',   # Sri Lanka
        '95': 'MM',   # Myanmar
        '98': 'IR',   # Iran
        '212': 'MA',  # Morocco
        '213': 'DZ',  # Algeria
        '216': 'TN',  # Tunisia
        '218': 'LY',  # Libya
        '220': 'GM',  # Gambia
        '221': 'SN',  # Senegal
        '234': 'NG',  # Nigeria
        '254': 'KE',  # Kenya
        '256': 'UG',  # Uganda
        '260': 'ZM',  # Zambia
        '263': 'ZW',  # Zimbabwe
        '351': 'PT',  # Portugal
        '352': 'LU',  # Luxembourg
        '358': 'FI',  # Finland
        '380': 'UA',  # Ukraine
        '420': 'CZ',  # Czech Republic
        '421': 'SK',  # Slovakia
        '886': 'TW',  # Taiwan
        '971': 'AE',  # United Arab Emirates
        '972': 'IL',  # Israel
        '977': 'NP',  # Nepal
        '998': 'UZ',  # Uzbekistan
    }

def is_valid_phone(value: Any) -> bool:
    """
    Validate if a value is a properly formatted phone number.

    Supports multiple formats including:
    - (country_code,operator_code,number,"comment")
    - +country_code-operator_code-number
    - (country_code) operator_code-number

    Parameters:
    -----------
    value : Any
        The value to validate

    Returns:
    --------
    bool
        True if the value is a valid phone number, False otherwise
    """
    if pd.isna(value) or not isinstance(value, str):
        return False

    # Strip whitespace
    phone = value.strip()

    # Match pattern (7,950,1234567,"comment") format
    pattern1 = r'^\([\d]+,[\d]+,[\d]+(,.*?)?\)$'

    # Match international format +7-950-1234567
    pattern2 = r'^\+?[\d]+[-\s]?[\d]+[-\s]?[\d]+'

    # Match (7) 950 1234567 format
    pattern3 = r'^\([\d]+\)\s*[\d][-\s]*[\d]'

    # Check against patterns
    if (re.match(pattern1, phone) or
            re.match(pattern2, phone) or
            re.match(pattern3, phone)):
        return True

    # Check if it's just digits (minimum 7 for a valid phone)
    if re.match(r'^[\d+() \-]{7,}$', phone):
        # Count digits
        digit_count = sum(c.isdigit() for c in phone)
        return digit_count >= 7

    return False


def load_messenger_patterns(csv_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Load messenger detection patterns from a CSV file.

    CSV format should have two columns: messenger_type and pattern

    Parameters:
    -----------
    csv_path : str, optional
        Path to the CSV file with patterns. If None, uses default patterns.

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping messenger types to regex patterns
    """
    patterns = DEFAULT_MESSENGER_PATTERNS.copy()

    if csv_path is not None and os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)

                # Skip header
                next(reader, None)

                # Read patterns
                for row in reader:
                    if len(row) >= 2:
                        messenger_type, pattern = row[0].lower(), row[1]

                        # Initialize if this messenger type is not in the dict
                        if messenger_type not in patterns:
                            patterns[messenger_type] = []

                        # Add pattern
                        patterns[messenger_type].append(pattern)

            logger.info(f"Loaded messenger patterns from {csv_path}")

        except Exception as e:
            logger.warning(f"Error loading messenger patterns from {csv_path}: {e}. Using default patterns.")

    return patterns


def detect_messenger_references(comment: str, patterns_csv: Optional[str] = None) -> Dict[str, bool]:
    """
    Detect references to messengers in a comment.

    Parameters:
    -----------
    comment : str
        The comment to analyze
    patterns_csv : str, optional
        Path to CSV with custom patterns. If None, uses default patterns.

    Returns:
    --------
    Dict[str, bool]
        Dictionary indicating presence of each messenger type
    """
    if pd.isna(comment) or not isinstance(comment, str) or not comment.strip():
        return {k: False for k in DEFAULT_MESSENGER_PATTERNS.keys()}

    # Load patterns
    patterns = load_messenger_patterns(patterns_csv)

    # Prepare comment for analysis
    text = comment.lower().strip()

    # Check for each messenger type
    result = {}
    for messenger_type, messenger_patterns in patterns.items():
        # Check if any pattern matches
        result[messenger_type] = any(
            re.search(pattern, text, re.IGNORECASE) is not None
            for pattern in messenger_patterns
        )

    return result

def parse_phone_number(phone: Any) -> Dict[str, Any]:
    """
    Parse a phone number into its components.

    Handles multiple formats including:
    - (country_code,operator_code,number,"comment")
    - +country_code-operator_code-number
    - (country_code) operator_code-number

    Parameters:
    -----------
    phone : Any
        The phone number to parse

    Returns:
    --------
    Dict[str, Any]
        Dictionary with parsed components or None if invalid
    """
    # Validate input type and handle non-string values
    if pd.isna(phone) or not isinstance(phone, str):
        return {
            'is_valid': False,
            'original': str(phone) if phone is not None else 'None',
            'error': 'Invalid input type'
        }

    # Clean up input by removing extra whitespace
    phone = phone.strip()

    # Split the phone string into the main phone part and any comments
    phone_part, comment = _split_phone_and_comment(phone)

    # Parse the phone number components (country code, operator code, etc.)
    components = _parse_phone_components(phone_part)

    if not components:
        return {
            'is_valid': False,
            'original': phone,
            'error': 'Could not extract phone from input'
        }

    results = {
        'original': phone,
        **components,
        'comment': comment,
        'messenger_mentions': detect_messenger_references(comment) if comment else {}
    }
    return results

def identify_country_code(phone: Any) -> Optional[str]:
    """
    Extract the country code from a phone number.

    Parameters:
    -----------
    phone : Any
        The phone number

    Returns:
    --------
    Optional[str]
        The country code, or None if not found
    """
    parsed = parse_phone_number(phone)
    if parsed and parsed.get('is_valid', False):
        return parsed.get('country_code', None)
    return None


def identify_operator_code(phone: Any, country_code: Optional[str] = None) -> Optional[str]:
    """
    Extract the operator code from a phone number.

    Parameters:
    -----------
    phone : Any
        The phone number
    country_code : str, optional
        Specific country code to match, if None any country is accepted

    Returns:
    --------
    Optional[str]
        The operator code, or None if not found
    """
    parsed = parse_phone_number(phone)
    if parsed and parsed.get('is_valid', False):
        # If country code is specified, check if it matches
        if country_code is not None:
            if parsed.get('country_code') == country_code:
                return parsed.get('operator_code', None)
            return None

        # Otherwise return operator code regardless of country
        return parsed.get('operator_code', None)

    return None


def normalize_phone(country_code: str, operator_code: str, number: str) -> Optional[str]:
    """
    Normalize a phone number to E.164 international format.

    Parameters:
    -----------
    phone : Any
        The phone number to normalize

    Returns:
    --------
    Optional[str]
        Normalized phone number or None if normalization fails
    """

    # Ensure country code starts with +
    if country_code and not country_code.startswith('+'):
        country_code = '+' + country_code

    # Build normalized number
    if country_code and operator_code and number:
        return f"{country_code}{operator_code}{number}"

    return None


def analyze_phone_field(df: pd.DataFrame,
                        field_name: str,
                        patterns_csv: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
    """
    Analyze a phone number field in the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    logger.info(f"Analyzing phone field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Basic statistics
    total_rows = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count

    # Parse and analyze phone components
    parsed_results = []
    country_codes = {}
    operator_codes = {}
    format_errors = []
    format_error_count = 0
    has_comment_count = 0
    normalization_success_count = 0
    has_extension_count = 0
    extension_examples= []
    messenger_mentions = {
        'telegram': 0,
        'whatsapp': 0,
        'viber': 0,
        'signal': 0,
        'wechat': 0,
        'line': 0,
        'other': 0
    }

    # Process each phone number
    for phone in df[field_name].dropna():
        parsed = parse_phone_number(phone)

        if parsed:
            parsed_results.append(parsed)

            if parsed.get('is_valid', False):
                # Try to normalize
                normalized = normalize_phone(
                    country_code=parsed.get("country_code", ""),
                    operator_code=parsed.get("operator_code", ""),
                    number=parsed.get("number", ""),
                )
                if normalized:
                    normalization_success_count += 1

                # Count country codes
                country_code = parsed.get('country_code', '')
                if country_code:
                    country_codes[country_code] = country_codes.get(country_code, 0) + 1

                # Count operator codes
                operator_code = parsed.get('operator_code', '')
                if operator_code and country_code:
                    # Store operator codes as country_code:operator_code
                    code_key = f"{country_code}:{operator_code}"
                    operator_codes[code_key] = operator_codes.get(code_key, 0) + 1

                if parsed.get('extension', ''):
                    has_extension_count += 1
                    extension_examples.append(parsed.get('extension', ''))

                # Count comments
                if parsed.get('comment', ''):
                    has_comment_count += 1

                    # Count messenger mentions
                    for messenger, mentioned in parsed.get('messenger_mentions', {}).items():
                        if mentioned:
                            messenger_mentions[messenger] = messenger_mentions.get(messenger, 0) + 1
            else:
                format_error_count += 1
                if len(format_errors) < 10:  # Limit the number of examples
                    format_errors.append(phone)

    # Create result stats
    stats = {
        'total_rows': total_rows,
        'null_count': int(null_count),
        'null_percentage': round((null_count / total_rows) * 100, 2) if total_rows > 0 else 0,
        'non_null_count': int(non_null_count),
        'valid_count': int(non_null_count - format_error_count),
        'valid_percentage': round(((non_null_count - format_error_count) / total_rows) * 100,
                                  2) if total_rows > 0 else 0,
        'normalization_success_count': int(normalization_success_count),
        'normalization_success_percentage': round((normalization_success_count / non_null_count) * 100,
                                                  2) if non_null_count > 0 else 0,
        'format_error_count': format_error_count,
        'has_comment_count': has_comment_count,
        'has_extension_count': int(has_extension_count),
        'country_codes': country_codes,
        'operator_codes': operator_codes,
        'messenger_mentions': messenger_mentions,
        'extension_examples': extension_examples,
    }

    # Add error examples if any
    if format_errors:
        stats['format_error_examples'] = format_errors

    return stats

def analyze_phone_chunk(chunk_df: pd.DataFrame, field_name: str,
                        patterns_csv: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """
    Analyze a phone number field in the given DataFrame.

    Parameters:
    -----------
    chunk_df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    logger.info(f"Analyzing chunk of {field_name}")

    # Initializing the result
    stats = {
        'total_rows': len(chunk_df),
        'null_count': chunk_df[field_name].isna().sum(),
        'non_null_count': len(chunk_df) - chunk_df[field_name].isna().sum(),
        'normalization_success_count': 0,
        'has_comment_count': 0,
        'has_extension_count': 0,
        'format_error_count': 0,
        'country_codes': {},
        'operator_codes': {},
        'messenger_mentions': {key: 0 for key in DEFAULT_MESSENGER_PATTERNS.keys()},
        'format_error_examples': [],
        'extension_examples': []
    }

    for phone in chunk_df[field_name].dropna():
        parsed = parse_phone_number(phone)

        if parsed:
            if parsed.get('is_valid', False):
                # Try to normalize
                normalized = normalize_phone(
                    country_code=parsed.get("country_code", ""),
                    operator_code=parsed.get("operator_code", ""),
                    number=parsed.get("number", ""),
                )
                if normalized:
                    stats['normalization_success_count'] += 1

                # Count country codes
                country_code = parsed.get('country_code', '')
                if country_code:
                    stats['country_codes'][country_code] = stats['country_codes'].get(country_code, 0) + 1

                # Count operator codes
                operator_code = parsed.get('operator_code', '')
                if operator_code and country_code:
                    # Store operator codes as country_code:operator_code
                    code_key = f"{country_code}:{operator_code}"
                    stats['operator_codes'][code_key] = stats['operator_codes'].get(code_key, 0) + 1

                if parsed.get('extension', ''):
                    stats['has_extension_count'] += 1
                    stats['extension_examples'].append(parsed.get('extension', ''))

                # Count comments
                if parsed.get('comment', ''):
                    stats['has_comment_count'] += 1

                    # Count messenger mentions
                    for messenger, mentioned in parsed.get('messenger_mentions', {}).items():
                        if mentioned:
                            stats['messenger_mentions'][messenger] = stats['messenger_mentions'].get(messenger, 0) + 1
            else:
                stats['format_error_count'] += 1
                stats['format_error_examples'].append(phone)

    return stats

def analyze_phone_field_with_chunk(df: pd.DataFrame, field_name: str, patterns_csv: Optional[str] = None,
                                    chunk_size: int = 10000, **kwargs) -> Dict[str, Any]:
    """
    Analyze a phone number field in the given DataFrame using chunk.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    chunk_size : int, optional
        Size of chunk
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    import joblib

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    results = []
    # Create chunks for processing
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    for chunk in chunks:
        chunk_result = analyze_phone_chunk(chunk, field_name, patterns_csv, **kwargs)
        if chunk_result:
            results.append(chunk_result)

    # Aggregate results from all chunks
    total_rows = sum(r['total_rows'] for r in results)
    null_count = sum(r['null_count'] for r in results)
    non_null_count = sum(r['non_null_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    normalization_success_count = sum(r['normalization_success_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    has_comment_count = sum(r['has_comment_count'] for r in results)
    has_extension_count = sum(r['has_extension_count'] for r in results)
    final_stats = {
        'total_rows': total_rows,
        'null_count': int(null_count),
        'null_percentage': round((null_count / total_rows) * 100.0, 2)
        if total_rows > 0 else 0,
        'non_null_count': int(non_null_count),
        'valid_count': int(non_null_count - format_error_count),
        'valid_percentage': round(((non_null_count - format_error_count) / total_rows) * 100, 2)
        if total_rows > 0 else 0,
        'normalization_success_count': int(normalization_success_count),
        'normalization_success_percentage': round((normalization_success_count / non_null_count) * 100, 2)
        if non_null_count > 0 else 0,
        'format_error_count': format_error_count,
        'has_comment_count': has_comment_count,
        'has_extension_count': int(has_extension_count),
        'country_codes': {k: sum(r['country_codes'].get(k, 0) for r in results) for k in
                          set(k for r in results for k in r['country_codes'].keys())},
        'operator_codes': {k: sum(r['operator_codes'].get(k, 0) for r in results) for k in
                           set(k for r in results for k in r['operator_codes'].keys())},
        'messenger_mentions': {k: sum(r['messenger_mentions'][k] for r in results)
                               for k in DEFAULT_MESSENGER_PATTERNS.keys()},
        'format_error_examples': [item for r in results for item in r['format_error_examples']],
        'extension_examples': [item for r in results for item in r.get('extension_examples', [])],
    }

    return final_stats

def analyze_phone_field_with_joblib(df: pd.DataFrame, field_name: str, patterns_csv: Optional[str] = None,
                                    n_jobs: int = -1, chunk_size: int = 10000, **kwargs) -> Dict[str, Any]:
    """
    Analyze a phone number field in the given DataFrame using Joblib.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    n_jobs : int, optional
        Number of worker
    chunk_size : int, optional
        Size of chunk
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    import joblib

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Create chunks for parallel processing (you could also create chunks of a specific size)
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Use Joblib's Parallel and delayed to process chunks in parallel
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(analyze_phone_chunk)(chunk, field_name, patterns_csv, **kwargs) for chunk in chunks)

    # Aggregate results from all chunks
    total_rows = sum(r['total_rows'] for r in results)
    null_count = sum(r['null_count'] for r in results)
    non_null_count = sum(r['non_null_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    normalization_success_count = sum(r['normalization_success_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    has_comment_count = sum(r['has_comment_count'] for r in results)
    has_extension_count = sum(r['has_extension_count'] for r in results)
    final_stats = {
        'total_rows': total_rows,
        'null_count': int(null_count),
        'null_percentage': round((null_count / total_rows) * 100.0, 2)
        if total_rows > 0 else 0,
        'non_null_count': int(non_null_count),
        'valid_count': int(non_null_count - format_error_count),
        'valid_percentage': round(((non_null_count - format_error_count) / total_rows) * 100, 2)
        if total_rows > 0 else 0,
        'normalization_success_count': int(normalization_success_count),
        'normalization_success_percentage': round((normalization_success_count / non_null_count) * 100, 2)
        if non_null_count > 0 else 0,
        'format_error_count': format_error_count,
        'has_comment_count': has_comment_count,
        'has_extension_count': int(has_extension_count),
        'country_codes': {k: sum(r['country_codes'].get(k, 0) for r in results) for k in
                          set(k for r in results for k in r['country_codes'].keys())},
        'operator_codes': {k: sum(r['operator_codes'].get(k, 0) for r in results) for k in
                           set(k for r in results for k in r['operator_codes'].keys())},
        'messenger_mentions': {k: sum(r['messenger_mentions'][k] for r in results)
                               for k in DEFAULT_MESSENGER_PATTERNS.keys()},
        'format_error_examples': [item for r in results for item in r['format_error_examples']],
        'extension_examples': [item for r in results for item in r.get('extension_examples', [])],
    }

    return final_stats

def analyze_phone_field_with_dask(df: pd.DataFrame, field_name: str, patterns_csv: Optional[str] = None,
                                  npartitions: int = 2, chunk_size: int = 10000, **kwargs) -> Dict[str, Any]:
    """
    Analyze a phone number field in the given DataFrame using Joblib.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    npartitions : int, optional
        Number of partitions
    chunk_size : int, optional
        Size of chunk
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    import dask.dataframe as dd

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Convert to Dask DataFrame
    dask_df = dd.from_pandas(df, npartitions=npartitions)

    # Apply the analysis to each partition (chunk)
    results = dask_df.map_partitions(analyze_phone_chunk, field_name, patterns_csv, **kwargs)

    # Compute the result (this triggers the parallel computation)
    results = results.compute()

    # Aggregate results from all chunks
    total_rows = sum(r['total_rows'] for r in results)
    null_count = sum(r['null_count'] for r in results)
    non_null_count = sum(r['non_null_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    normalization_success_count = sum(r['normalization_success_count'] for r in results)
    format_error_count = sum(r['format_error_count'] for r in results)
    has_comment_count = sum(r['has_comment_count'] for r in results)
    has_extension_count = sum(r['has_extension_count'] for r in results)
    final_stats = {
        'total_rows': total_rows,
        'null_count': int(null_count),
        'null_percentage': round((null_count / total_rows) * 100.0, 2)
        if total_rows > 0 else 0,
        'non_null_count': int(non_null_count),
        'valid_count': int(non_null_count - format_error_count),
        'valid_percentage': round(((non_null_count - format_error_count) / total_rows) * 100, 2)
        if total_rows > 0 else 0,
        'normalization_success_count': int(normalization_success_count),
        'normalization_success_percentage': round((normalization_success_count / non_null_count) * 100, 2)
        if non_null_count > 0 else 0,
        'format_error_count': format_error_count,
        'has_comment_count': has_comment_count,
        'has_extension_count': int(has_extension_count),
        'extension_examples': [item for r in results for item in r.get('extension_examples', [])],
        'country_codes': {k: sum(r['country_codes'].get(k, 0) for r in results) for k in
                          set(k for r in results for k in r['country_codes'].keys())},
        'operator_codes': {k: sum(r['operator_codes'].get(k, 0) for r in results) for k in
                           set(k for r in results for k in r['operator_codes'].keys())},
        'messenger_mentions': {k: sum(r['messenger_mentions'][k] for r in results)
                               for k in DEFAULT_MESSENGER_PATTERNS.keys()},
        'format_error_examples': [item for r in results for item in r['format_error_examples']],
        'extension_examples': [item for r in results for item in r.get('extension_examples', [])],
    }

    return final_stats


def create_country_code_dictionary(
    df: pd.DataFrame, field_name: str, min_count: int = 1
) -> Dict[str, Any]:
    """
    Create a frequency dictionary for country codes.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the phone field
    min_count : int
        Minimum frequency for inclusion in the dictionary
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    Dict[str, Any]
        Dictionary with country code frequency data and metadata
    """
    logger.info(f"Creating country code dictionary for phone field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    try:
        # Extract country codes
        country_codes = {}

        for phone in df[field_name].dropna():
            country_code = identify_country_code(phone)
            if country_code:
                country_codes[country_code] = country_codes.get(country_code, 0) + 1

        # Filter by minimum count
        filtered_codes = {code: count for code, count in country_codes.items() if count >= min_count}

        # Convert to list format for CSV output
        code_list = []
        for code, count in filtered_codes.items():
            code_list.append({
                'country_code': code,
                'count': count,
                'percentage': 0  # Will calculate below
            })

        # Add percentages
        total_phones = sum(filtered_codes.values())
        for item in code_list:
            item['percentage'] = round((item['count'] / total_phones) * 100, 2) if total_phones > 0 else 0

        # Sort by count (descending)
        code_list = sorted(code_list, key=lambda x: x['count'], reverse=True)

        return {
            'field_name': field_name,
            'total_country_codes': len(code_list),
            'total_phones': total_phones,
            'country_codes': code_list
        }

    except Exception as e:
        logger.error(f"Error creating country code dictionary for {field_name}: {e}", exc_info=True)
        return {'error': str(e)}

def create_operator_code_dictionary(df: pd.DataFrame,
                                    field_name: str,
                                    country_codes: Optional[List[str]] = None,
                                    min_count: int = 1,
                                    **kwargs) -> Dict[str, Any]:
    """
    Create a frequency dictionary for operator codes.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the phone field
    country_code : List[str] or str, optional
        The country code(s) to filter by (if None, use all)
    min_count : int
        Minimum frequency for inclusion in the dictionary
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    Dict[str, Any]
        Dictionary with operator code frequency data and metadata
    """
    logger.info(f"Creating operator code dictionary for phone field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    try:
        # Extract operator codes
        operator_codes = {}
        # Normalize country codes input
        normalized_country_codes = []
        
        if country_codes is not None:
            # Ensure we have a list
            if isinstance(country_codes, str):
                country_codes = [country_codes]
            
            # Convert any ISO country codes to numeric format
            for code in country_codes:
                if code and not str(code).isdigit():
                    try:
                        # Convert ISO country code (e.g. 'FR') to numeric code ('33')
                        numeric_code = str(phonenumbers.country_code_for_region(code))
                        normalized_country_codes.append(numeric_code)
                    except Exception:
                        logger.warning(f"Could not convert ISO country code {code} to numeric format")
                else:
                    normalized_country_codes.append(code)
        
        for phone in df[field_name].dropna():
            parsed = parse_phone_number(phone)

            if parsed and parsed.get('is_valid', False):
                parsed_country_code = parsed.get('country_code', '')
                operator_code = parsed.get('operator_code', '')

                # Skip if no operator code
                if not operator_code:
                    continue
                
                # Check country code match
                if country_codes is not None:
                    # Check if parsed country code matches any in our list (either original or numeric)
                    if (parsed_country_code in normalized_country_codes):
                        key = operator_code
                    else:
                        continue
                else:
                    key = f"{parsed_country_code}:{operator_code}"

                operator_codes[key] = operator_codes.get(key, 0) + 1

        # Filter by minimum count
        filtered_codes = {code: count for code, count in operator_codes.items() if count >= min_count}

        # Convert to list format for CSV output
        code_list = []
        for code, count in filtered_codes.items():
            # Extract parts if using country:operator format
            if ':' in code and country_codes is None:
                parts = code.split(':')
                country = parts[0]
                operator = parts[1]
                code_list.append({
                    'country_code': country,
                    'operator_code': operator,
                    'count': count,
                    'percentage': 0  # Will calculate below
                })
            else:
                code_list.append({
                    'operator_code': code,
                    'count': count,
                    'percentage': 0  # Will calculate below
                })

        # Add percentages
        total_phones = sum(filtered_codes.values())
        for item in code_list:
            item['percentage'] = round((item['count'] / total_phones) * 100, 2) if total_phones > 0 else 0

        # Sort by count (descending)
        code_list = sorted(code_list, key=lambda x: x['count'], reverse=True)

        return {
            'field_name': field_name,
            'country_code': country_codes,  # Original input
            'total_operator_codes': len(code_list),
            'total_phones': total_phones,
            'operator_codes': code_list
        }

    except Exception as e:
        logger.error(f"Error creating operator code dictionary for {field_name}: {e}", exc_info=True)
        return {'error': str(e)}


def create_messenger_dictionary(
    df: pd.DataFrame,
    field_name: str,
    min_count: int = 1,
    patterns_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a frequency dictionary for messenger mentions in phone comments.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the phone field
    min_count : int
        Minimum frequency for inclusion in the dictionary
    patterns_csv : str, optional
        Path to CSV with messenger patterns
    **kwargs : dict
        Additional parameters

    Returns:
    --------
    Dict[str, Any]
        Dictionary with messenger frequency data and metadata
    """
    logger.info(f"Creating messenger dictionary for phone field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    try:
        # Extract messenger mentions
        messenger_counts = {
            'telegram': 0,
            'whatsapp': 0,
            'viber': 0,
            'signal': 0,
            'wechat': 0,
            'line': 0,
            'other': 0
        }

        for phone in df[field_name].dropna():
            parsed = parse_phone_number(phone)
            if parsed and parsed.get('is_valid', False) and parsed.get('comment', ''):
                comment = parsed.get('comment', '')

                # Detect messenger references
                mentions = detect_messenger_references(comment, patterns_csv)

                # Count mentions
                for messenger, mentioned in mentions.items():
                    if mentioned:
                        messenger_counts[messenger] = messenger_counts.get(messenger, 0) + 1

        # Filter by minimum count
        filtered_messengers = {m: count for m, count in messenger_counts.items() if count >= min_count}

        # Convert to list format for CSV output
        messenger_list = []
        for messenger, count in filtered_messengers.items():
            messenger_list.append({
                'messenger': messenger,
                'count': count,
                'percentage': 0  # Will calculate below
            })

        # Add percentages
        total_mentions = sum(filtered_messengers.values())
        for item in messenger_list:
            item['percentage'] = round((item['count'] / total_mentions) * 100, 2) if total_mentions > 0 else 0

        # Sort by count (descending)
        messenger_list = sorted(messenger_list, key=lambda x: x['count'], reverse=True)

        return {
            'field_name': field_name,
            'total_messenger_types': len(messenger_list),
            'total_mentions': total_mentions,
            'messengers': messenger_list
        }

    except Exception as e:
        logger.error(f"Error creating messenger dictionary for {field_name}: {e}", exc_info=True)
        return {'error': str(e)}


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for analyzing the phone field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field to analyze

    Returns:
    --------
    Dict[str, Any]
        Estimated resource requirements
    """
    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Get basic data about the field
    total_rows = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count

    # Sample a subset of values for quicker analysis
    sample_size = min(1000, non_null_count)
    if non_null_count > 0:
        sample = df[field_name].dropna().sample(n=sample_size, random_state=42)
    else:
        sample = pd.Series([])

    # Test parsing on sample
    valid_count = 0
    comment_count = 0
    countries = set()
    operators = set()

    for phone in sample:
        parsed = parse_phone_number(phone)
        if parsed and parsed.get('is_valid', False):
            valid_count += 1

            if parsed.get('comment', ''):
                comment_count += 1

            country = parsed.get('country_code', '')
            if country:
                countries.add(country)

            operator = parsed.get('operator_code', '')
            if operator:
                operators.add(operator)

    # Calculate validity rate
    validity_rate = valid_count / len(sample) if len(sample) > 0 else 0
    comment_rate = comment_count / valid_count if valid_count > 0 else 0

    # Extrapolate to full dataset
    estimated_valid = int(non_null_count * validity_rate)
    estimated_with_comments = int(estimated_valid * comment_rate)

    # Estimate unique country and operator codes
    estimated_countries = len(countries)
    estimated_operators = len(operators)

    # Estimate memory requirements (rough approximation)
    avg_phone_length = sample.astype(str).str.len().mean() if len(sample) > 0 else 0

    # Base memory for dataframe
    base_memory_mb = (non_null_count * avg_phone_length * 2) / (1024 * 1024)

    # Memory for parsed components
    parsed_memory_mb = (estimated_valid * 500) / (1024 * 1024)  # ~500 bytes per parsed record

    # Memory for dictionaries and stats
    dict_memory_mb = (estimated_countries * 100 + estimated_operators * 100) / (1024 * 1024)

    # Total memory estimation
    total_memory_mb = base_memory_mb + parsed_memory_mb + dict_memory_mb

    # Estimate processing time based on data volume
    processing_time_sec = 0.1 + (non_null_count * 0.0001)  # Base + per record time

    # Adjust time if many comments to analyze
    if estimated_with_comments > 1000:
        processing_time_sec += estimated_with_comments * 0.0002

    return {
        'total_rows': total_rows,
        'non_null_count': int(non_null_count),
        'estimated_valid_phones': estimated_valid,
        'estimated_with_comments': estimated_with_comments,
        'estimated_unique_country_codes': estimated_countries,
        'estimated_unique_operator_codes': estimated_operators,
        'estimated_memory_mb': round(total_memory_mb, 2),
        'estimated_processing_time_sec': round(processing_time_sec, 2)
    }


def _parse_by_regex(phone: str, extension: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse phone number using common regex patterns.
    Supports formats: +country-operator-number or (country) operator-number.

    Parameters:
    -----------
    phone : str
        Cleaned phone string.
    extension : str or None
        Phone extension if present.

    Returns:
    --------
    dict or None
        Parsed components or None if not matched.
    """
    # Try NANP/US format
    match1 = re.match(PhonePatterns.PHONE_REGEX_PATTERNS["TEN_DIGIT"], phone)
    if match1:
        return {
            "country_code": _detect_country_code(phone),
            "operator_code": match1.group(1).replace("(", "").replace(")", "").strip(),
            "number": f"{match1.group(2)}-{match1.group(3)}",
            "extension": extension,
            "is_valid": True,
        }

    # Try international format
    match2 = re.match(PhonePatterns.PHONE_REGEX_PATTERNS["INTERNATIONAL"], phone)
    if match2:
        return {
            "country_code": match2.group(1),
            "operator_code": match2.group(2),
            "number": match2.group(3),
            "extension": extension,
            "is_valid": True,
        }

    # Try parentheses format
    match3 = re.match(PhonePatterns.PHONE_REGEX_PATTERNS["PARENTHESES"], phone)
    if match3:
        return {
            "country_code": match3.group(1),
            "operator_code": match3.group(2),
            "number": match3.group(3),
            "extension": extension,
            "is_valid": True,
        }

    return None


def _parse_by_phonenumbers(
    phone: str, extension: Optional[str], detected_region: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Parse phone number using the phonenumbers library.
    Tries detected region first, then None.

    Parameters:
    -----------
    phone_ : str
        Phone string.
    extension : str or None
        Phone extension if present.
    detected_region : str or None
        Region code detected from phone.

    Returns:
    --------
    dict or None
        Parsed components or None if not valid.
    """
    try:
        parsed = None
        # Strategy 1: Try with detected region
        if detected_region:
            try:
                temp_parsed = phonenumbers.parse(phone, detected_region)
                if phonenumbers.is_possible_number(temp_parsed):
                    parsed = temp_parsed
            except Exception as e:
                logger.debug(f"Phonenumbers region parse failed: {e}")

        # Strategy 2: Try with None region
        if parsed is None:
            try:
                parsed = phonenumbers.parse(phone, None)
            except Exception as e:
                logger.debug(f"Phonenumbers None region parse failed: {e}")

        # Final validation
        if parsed is None or not phonenumbers.is_possible_number(parsed):
            return None

        country_code = str(parsed.country_code)
        national_number = str(parsed.national_number)
        ndc_len = phonenumbers.length_of_national_destination_code(parsed)
        operator_code = national_number[:ndc_len] if ndc_len > 0 else ""
        number = national_number[ndc_len:] if ndc_len > 0 else national_number

        return {
            "country_code": country_code,
            "operator_code": operator_code,
            "number": number,
            "extension": extension,
            "is_valid": True,
        }
    except Exception as e:
        logger.debug(f"Phonenumbers parsing failed: {e}")
        return None

def _detect_country_code(phone: str) -> Optional[str]:
    """
    Detect the country code from a phone number string using the phonenumbers library.

    This function takes a phone number string and attempts to extract its country code
    using multiple strategies to handle various formats and representations.

    Parameters:
    -----------
    phone : str
        The phone number string from which to extract the country code.
        Can be in various formats (E.164, national, with/without formatting).

    Returns:
    --------
    Optional[str]
        The extracted country code as a string (without '+' prefix), or None if detection fails.
    """
    if not phone or not isinstance(phone, str):
        return None

    phone = phone.strip()
    if not phone:
        return None

    # Special case for North American format: NXX-NXX-XXXX where N is 2-9
    na_pattern = re.match(r"^([2-9]\d{2})[- ]?(\d{3})[- ]?(\d{4})$", phone)
    if na_pattern:
        return "1"  # North American Numbering Plan

    # Strategy 1: Try direct parsing with phonenumbers
    try:
        # If it starts with +, parse with None region
        if "+" in phone:
            parsed = phonenumbers.parse(phone, None)
            if phonenumbers.is_possible_number(parsed):
                return str(parsed.country_code)
    except Exception:
        pass  # Continue to next method if this fails

    # Strategy 2: Try with normalized format
    try:
        # Convert 00XX to +XX format for international numbers
        normalized = _normalize_phone_number(phone)

        # Try adding + for likely international numbers
        if not normalized.startswith("+") and len(normalized) >= 10:
            try:
                alt_normalized = "+" + normalized
                parsed = phonenumbers.parse(alt_normalized, None)
                if phonenumbers.is_possible_number(parsed):
                    return str(parsed.country_code)
            except Exception:
                pass

        # Try parsing the normalized number
        parsed = phonenumbers.parse(normalized, None)
        if phonenumbers.is_possible_number(parsed):
            return str(parsed.country_code)
    except Exception:
        pass  # Continue to next method if this fails

    # Strategy 3: Try with common regions as hints
    # Start with US for NANP-formatted numbers
    if re.match(r"^\(?[2-9]\d{2}\)?[- ]?\d{3}[- ]?\d{4}$", phone):
        try:
            parsed = phonenumbers.parse(phone, "US")
            if phonenumbers.is_possible_number(parsed):
                return str(parsed.country_code)
        except Exception:
            pass

    # Try other common regions
    for region in ["US", "CA", "GB", "FR", "DE", "RU", "CN", "IN", "BR", "JP"]:
        try:
            parsed = phonenumbers.parse(phone, region)
            if phonenumbers.is_possible_number(parsed):
                return str(parsed.country_code)
        except Exception:
            continue

    # Strategy 4: Check for common country code patterns
    # Strip all non-digit characters except leading +
    cleaned = re.sub(r"[^\d+]", "", phone)

    # Check for NANP pattern (without prefix but with correct format)
    if len(cleaned) == 10 and cleaned[0] in "23456789":
        return "1"

    # Handle numbers starting with + followed by 1-3 digits as country code
    if cleaned.startswith("+"):
        # Extract potential country code (1-3 digits)
        for i in range(1, 4):
            if len(cleaned) > i:
                potential_cc = cleaned[1 : i + 1]
                try:
                    # Try to validate it's a real country code
                    for region in phonenumbers.SUPPORTED_REGIONS:
                        if phonenumbers.country_code_for_region(region) == int(
                            potential_cc
                        ):
                            return potential_cc
                except Exception:
                    continue

    # Handle numbers starting with 00 followed by 1-3 digits as country code
    elif cleaned.startswith("00"):
        # Extract potential country code (1-3 digits)
        for i in range(2, 5):
            if len(cleaned) > i:
                potential_cc = cleaned[2 : i + 1]
                try:
                    # Try to validate it's a real country code
                    for region in phonenumbers.SUPPORTED_REGIONS:
                        if phonenumbers.country_code_for_region(region) == int(
                            potential_cc
                        ):
                            return potential_cc
                except Exception:
                    continue

    # Failed to detect country code
    return None

def _fallback_split(
    phone: str, extension: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Fallback manual split for phone numbers when other methods fail.
    Only for digit-only strings, tries common country/operator code lengths.

    Parameters:
    -----------
    phone : str
        Phone string.
    extension : str or None
        Phone extension if present.

    Returns:
    --------
    dict or None
        Parsed components or None if not valid.
    """
    digits_only = re.sub(r"[^\d]", "", phone)
    if len(digits_only) >= 7:
        # Try country code lengths (1-3 digits)
        for cc_len in [1, 2, 3]:
            if (
                len(digits_only) > cc_len + 6
            ):  # Need at least 6 digits for operator + number
                country_code = digits_only[:cc_len]
                remaining = digits_only[cc_len:]

                # Try operator code lengths (2-4 digits)
                for op_len in [2, 3, 4]:
                    if (
                        len(remaining) >= op_len + 4
                    ):  # Need at least 4 digits for number
                        operator_code = remaining[:op_len]
                        number = remaining[op_len:]
                        return {
                            "country_code": country_code,
                            "operator_code": operator_code,
                            "number": number,
                            "extension": extension,
                            "is_valid": _detect_country_code(country_code) is not None,
                        }
    return None


def _split_phone_and_comment(phone_str: str) -> Tuple[str, str]:
    """
    Split phone string into phone part and comment part.
    Parameters:
    -----------
    phone_str : str
        The input string containing phone number and potentially a comment.
        Can be in various formats with different separators.

    Returns:
    --------
    Tuple[str, str]
        A tuple containing:
        - phone (str): The extracted phone number part
        - comment (str): The extracted comment part, empty string if no comment found

    Examples:
    - "447529138400 via Viber"       -> ("447529138400", "via Viber")
    - "(7,950,1234567,\"viber\")"    -> ("(7,950,1234567,\"viber\")", "")
    - "+33 6 70 70 40 39, telegram"  -> ("+33 6 70 70 40 39", "telegram")
    - "33 6 12 34 56 78"             -> ("33 6 12 34 56 78", "")
    """
    if not isinstance(phone_str, str):
        return "", ""

    phone_str = phone_str.strip()

    regex = r'(?P<phone>\+?\d[\d\s()\-.]{6,}|\(\d+,\d+,\d+(?:,\s*"?[^)]+"?)?\))[\s,;:]+(?P<comment>[A-Za-z"\'].+)'

    match = re.search(regex, phone_str)
    if match:
        phone = match.group("phone").strip()
        comment = match.group("comment").strip() if match.group("comment") else ""
        return phone, comment

    return phone_str, ""

def _extract_extension(phone_str: str) -> Tuple[str, Optional[str]]:
    """
    Extracts a phone extension from a phone number string if present.

    Parameters:
    -----------
    phone_str : str
        The input phone number string, which may contain an extension in various formats
        (e.g., "ext123", "x123", "poste 123", etc.).

    Returns:
    --------
    Tuple[str, Optional[str]]
        - The cleaned phone number string with the extension removed
        - The extracted extension as a string, or None if no extension is found
    """
    if not phone_str or not isinstance(phone_str, str):
        return "", None

    phone_str = phone_str.strip()
    if not phone_str:
        return "", None

    for pattern in PhonePatterns.EXTENSION_PATTERNS:
        match = re.search(pattern, phone_str, flags=re.IGNORECASE)
        if match:
            extension = match.group(1)
            # Validate the extension (must be all digits, max 6 characters)
            if len(extension) <= 6 and extension.isdigit():
                # Remove the extension from the original string
                cleaned_phone = re.sub(pattern, '', phone_str, flags=re.IGNORECASE).strip()
                # Remove trailing commas or spaces
                cleaned_phone = re.sub(r'[,\s]+$', '', cleaned_phone)
                return cleaned_phone, extension

    return phone_str, None

def _normalize_scientific_notation_string(s: Any) -> Any:
    """
    Convert a string in scientific notation (e.g., '1.234e+10') to a plain integer string.

    Parameters
    ----------
    s : Any
        The input value, expected to be a string or number that may be in scientific notation.

    Returns
    -------
    str or original type
        If conversion is possible, returns the integer as a string (e.g., '12340000000').
        If not, returns the original input unchanged.

    Notes
    -----
    - This function is useful for cleaning phone numbers that may have been read from
      spreadsheets or CSV files and were automatically converted to scientific notation.
    - If the input is not a valid number in scientific notation, it is returned as-is.
    """
    try:
        f = float(s)
        return str(int(f))
    except:
        return s
        
def _normalize_phone_number(phone: str) -> str:
    """
    Normalize international phone number prefix from 00 to + format.
    
    This function standardizes international phone number prefixes by converting
    the international access code "00" to the standard "+" format used in E.164
    international numbering plan.
    
    Parameters:
    -----------
    phone : str
        The phone number string to normalize. Can contain various formats
        including international prefixes like "00" or "+"
        
    Returns:
    --------
    str
        The normalized phone number with "00" prefix converted to "+" format
        
    Examples:
    ---------
    >>> normalize_phone_number("0033612345678")
    "+33612345678"
    >>> normalize_phone_number("+33612345678")
    "+33612345678"
    >>> normalize_phone_number("612345678")
    "612345678"
    
    Notes:
    ------
    - Only handles "00" to "+" conversion, does not perform full E.164 normalization
    - Preserves original format if it doesn't start with "00"
    - Strips leading/trailing whitespace before processing
    """
    # Remove leading and trailing whitespace
    phone = phone.strip()
    
    # Convert scientific notation string to integer string if needed
    phone = _normalize_scientific_notation_string(phone)

    # Remove formatting characters
    phone = re.sub(r"[()\s\-]", "", phone)
    
    # Convert international access code "00" to standard "+" prefix
    if phone.startswith("00"):
        phone = "+" + phone[2:]
    
    return phone

def _detect_region_for_phone(phone: str) -> Optional[str]:
    """
    Detect the most likely region (country) for a phone number.
    
    Parameters:
    -----------
    phone : str
        Cleaned phone number string (can be with or without country code)
        
    Returns:
    --------
    Optional[str]
        ISO country code (e.g., 'US', 'RU', 'FR') or None if detection fails
    """
    if not phone or not isinstance(phone, str):
        return None
        
    phone = phone.strip()
    if not phone:
        return None
        
    # First attempt: Try using the phonenumbers library directly
    try:
        # If it starts with +, parse with None region
        if '+' in phone:
            parsed = phonenumbers.parse(phone, None)
            if phonenumbers.is_possible_number(parsed):
                return phonenumbers.region_code_for_number(parsed)
    except Exception:
        pass  # Continue to next method if this fails
    
    # Second attempt: Convert to E.164 format and try again
    try:
        # Remove all non-digit characters except leading +
        normalized = re.sub(r'[^\d+]', '', phone)
        
        # Add + if not present and number is long enough to have country code
        if not normalized.startswith('+') and len(normalized) >= 10:
            normalized = '+' + normalized
            
        parsed = phonenumbers.parse(normalized, None)
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.region_code_for_number(parsed)
    except Exception:
        pass  # Continue to fallback method
    
    # Third attempt: Fallback to manual detection for common patterns
    digits_only = re.sub(r'[^\d]', '', phone)
    
    # Special case for North American Numbering Plan (NANP)
    if len(digits_only) == 10 and digits_only[0] in '23456789':
        return 'US'  # Default to US for 10-digit numbers
    
    # Try different country code lengths (1-3 digits)
    for prefix_len in [3, 2, 1]:
        if len(digits_only) >= prefix_len:
            prefix = digits_only[:prefix_len]
            if prefix in COUNTRY_PREFIX_MAP:
                return COUNTRY_PREFIX_MAP[prefix]
    
    # If all else fails, try one more time with phonenumbers library
    try:
        # Try with possible region hint of 'US' as fallback
        parsed = phonenumbers.parse(phone, 'US')
        if phonenumbers.is_possible_number(parsed):
            return phonenumbers.region_code_for_number(parsed)
    except Exception:
        pass
        
    return None

def _parse_phone_components(phone: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Parse a phone string into its main components: country_code, operator_code, number, and extension.
    
    This function implements a multi-strategy approach for parsing phone numbers:
    1. Validates and cleans the input
    2. Attempts pattern-based parsing with regex
    3. Normalizes and detects region
    4. Uses phonenumbers library for advanced parsing
    5. Falls back to heuristic parsing if needed
    
    Parameters:
    -----------
    phone : str
        The phone number string to parse
        
    Returns:
    --------
    Optional[Dict[str, Optional[str]]]
        Dictionary with phone components or None if parsing fails
    """
    # Input validation
    if not phone or not isinstance(phone, str):
        return None
    phone = phone.strip()
    if not phone:
        return None
        
    # Step 1: Extract extension if present
    phone_clean, extension = _extract_extension(phone)
    
    # Step 2: Try regex parsing first (fastest method)
    result = _parse_by_regex(phone_clean, extension)
    if result:
        return result
        
    # Step 3: Normalize and prepare for advanced parsing
    phone_clean = _normalize_phone_number(phone_clean)
    original_for_detection = phone_clean
    
    # Step 4: Detect region for better accuracy
    detected_region = _detect_region_for_phone(original_for_detection)
    
    # Step 5: Ensure proper international format
    if not phone_clean.startswith("+"):
        digits_only = "".join(c for c in phone_clean if c.isdigit())
        if detected_region == 'US' and len(digits_only) == 10:
            phone_clean = "+1" + digits_only
        else:
            phone_clean = "+" + digits_only
            
    # Step 6: Try parsing with phonenumbers library (most accurate)
    result = _parse_by_phonenumbers(phone_clean, extension, detected_region)
    if result:
        return result
        
    # Step 7: Fallback to manual split as last resort
    result = _fallback_split(phone_clean, extension)
    if result:
        return result
        
    return None
