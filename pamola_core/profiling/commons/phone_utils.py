"""
Phone number analysis utilities for the HHR anonymization project.

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
from typing import Dict, List, Any, Optional

import pandas as pd

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


def parse_phone_number(phone: Any) -> Optional[Dict[str, Any]]:
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
    Optional[Dict[str, Any]]
        Dictionary with parsed components or None if invalid
    """
    if pd.isna(phone) or not isinstance(phone, str):
        return None

    phone = phone.strip()

    # Try to match the format (7,950,1234567,"comment")
    pattern1 = r'^\((\d+),(\d+),(\d+)(?:,\s*\"?(.*?)\"?)?\)$'
    match1 = re.match(pattern1, phone)

    if match1:
        country_code = match1.group(1)
        operator_code = match1.group(2)
        number = match1.group(3)
        comment = match1.group(4) if match1.group(4) else ""

        return {
            'is_valid': True,
            'original': phone,
            'country_code': country_code,
            'operator_code': operator_code,
            'number': number,
            'comment': comment,
            'messenger_mentions': detect_messenger_references(comment)
        }

    # Try to match +7-950-1234567 format
    pattern2 = r'^\+?(\d+)[-\s]?(\d{3,4})[-\s]?(\d+)$'
    match2 = re.match(pattern2, phone)

    if match2:
        country_code = match2.group(1)
        operator_code = match2.group(2)
        number = match2.group(3)

        return {
            'is_valid': True,
            'original': phone,
            'country_code': country_code,
            'operator_code': operator_code,
            'number': number,
            'comment': "",
            'messenger_mentions': {}
        }

    # Try to match (7) 950 1234567 format
    pattern3 = r'^\((\d+)\)\s*(\d{3,4})[-\s]*(\d+)$'
    match3 = re.match(pattern3, phone)

    if match3:
        country_code = match3.group(1)
        operator_code = match3.group(2)
        number = match3.group(3)

        return {
            'is_valid': True,
            'original': phone,
            'country_code': country_code,
            'operator_code': operator_code,
            'number': number,
            'comment': "",
            'messenger_mentions': {}
        }

    # Try to extract just the digits and guess components
    digits = ''.join(c for c in phone if c.isdigit())

    if len(digits) >= 7:
        # Guess country code (1-3 digits)
        country_code = digits[0:1]  # Default to first digit
        if digits.startswith('7') or digits.startswith('1'):
            country_code = digits[0:1]
            rest = digits[1:]
        elif digits.startswith('33') or digits.startswith('44') or digits.startswith('86'):
            country_code = digits[0:2]
            rest = digits[2:]
        elif digits.startswith('375') or digits.startswith('380'):
            country_code = digits[0:3]
            rest = digits[3:]
        else:
            # Unknown format, make best guess
            country_code = digits[0:1]
            rest = digits[1:]

        # Guess operator code (usually 3-4 digits)
        if len(rest) >= 4:
            operator_code = rest[0:3]
            number = rest[3:]
        else:
            operator_code = ""
            number = rest

        return {
            'is_valid': True,
            'original': phone,
            'country_code': country_code,
            'operator_code': operator_code,
            'number': number,
            'comment': "",
            'messenger_mentions': {}
        }

    # If all parsing attempts fail
    return {
        'is_valid': False,
        'original': phone,
        'error': 'Could not parse phone number'
    }


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


def normalize_phone(phone: Any) -> Optional[str]:
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
    parsed = parse_phone_number(phone)
    if not parsed or not parsed.get('is_valid', False):
        return None

    country_code = parsed.get('country_code', '')
    operator_code = parsed.get('operator_code', '')
    number = parsed.get('number', '')

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
                normalized = normalize_phone(phone)
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
        'country_codes': country_codes,
        'operator_codes': operator_codes,
        'messenger_mentions': messenger_mentions
    }

    # Add error examples if any
    if format_errors:
        stats['format_error_examples'] = format_errors

    return stats


def create_country_code_dictionary(df: pd.DataFrame,
                                   field_name: str,
                                   min_count: int = 1,
                                   **kwargs) -> Dict[str, Any]:
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
                                    country_code: Optional[str] = None,
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
    country_code : str, optional
        The country code to filter by (if None, use all)
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

        for phone in df[field_name].dropna():
            parsed = parse_phone_number(phone)

            if parsed and parsed.get('is_valid', False):
                parsed_country_code = parsed.get('country_code', '')
                operator_code = parsed.get('operator_code', '')

                # Skip if no operator code or if it doesn't match country filter
                if not operator_code:
                    continue

                if country_code is not None and parsed_country_code != country_code:
                    continue

                # Use operator code directly if country filter specified,
                # otherwise use country:operator format
                if country_code is not None:
                    key = operator_code
                else:
                    key = f"{parsed_country_code}:{operator_code}"

                operator_codes[key] = operator_codes.get(key, 0) + 1

        # Filter by minimum count
        filtered_codes = {code: count for code, count in operator_codes.items() if count >= min_count}

        # Convert to list format for CSV output
        code_list = []
        for code, count in filtered_codes.items():
            # Extract parts if using country:operator format
            if ':' in code and country_code is None:
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
            'country_code': country_code,  # Will be None if not filtered
            'total_operator_codes': len(code_list),
            'total_phones': total_phones,
            'operator_codes': code_list
        }

    except Exception as e:
        logger.error(f"Error creating operator code dictionary for {field_name}: {e}", exc_info=True)
        return {'error': str(e)}


def create_messenger_dictionary(df: pd.DataFrame,
                                field_name: str,
                                min_count: int = 1,
                                patterns_csv: Optional[str] = None,
                                **kwargs) -> Dict[str, Any]:
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