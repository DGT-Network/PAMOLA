"""
Date analysis utilities for the project.

This module provides utility functions for analyzing date fields, including
validation, distribution analysis, anomaly detection, and group analysis.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def prepare_date_data(df: pd.DataFrame, field_name: str) -> Tuple[pd.Series, int, int]:
    """
    Prepare date data for analysis.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze

    Returns:
    --------
    Tuple[pd.Series, int, int]
        Tuple containing the prepared date series, null count, and non-null count
    """
    # Check if field exists
    if field_name not in df.columns:
        raise ValueError(f"Field {field_name} not found in DataFrame")

    # Get basic counts
    total_records = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_records - null_count

    # Convert to datetime for analysis
    dates = pd.to_datetime(df[field_name], errors='coerce')

    return dates, null_count, non_null_count


def calculate_date_stats(dates: pd.Series) -> Dict[str, Any]:
    """
    Calculate basic date statistics.

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze

    Returns:
    --------
    Dict[str, Any]
        Dictionary with date statistics
    """
    # Skip if no valid dates
    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()

    if valid_count == 0:
        return {
            'valid_count': 0,
            'invalid_count': 0,
            'min_date': None,
            'max_date': None
        }

    # Calculate date range
    min_date = dates[valid_mask].min()
    max_date = dates[valid_mask].max()

    return {
        'valid_count': int(valid_count),
        'min_date': min_date.strftime('%Y-%m-%d') if not pd.isna(min_date) else None,
        'max_date': max_date.strftime('%Y-%m-%d') if not pd.isna(max_date) else None
    }


def calculate_distributions(dates: pd.Series) -> Dict[str, Dict[str, int]]:
    """
    Calculate various date distributions (year, decade, month, day of week).

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze

    Returns:
    --------
    Dict[str, Dict[str, int]]
        Dictionary with various date distributions
    """
    # Skip if no valid dates
    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()

    if valid_count == 0:
        return {}

    result = {}

    # Year distribution
    year_distribution = dates[valid_mask].dt.year.value_counts().sort_index().to_dict()
    result['year_distribution'] = {str(year): int(count) for year, count in year_distribution.items()}

    # Decade distribution
    decades = (dates[valid_mask].dt.year // 10 * 10).value_counts().sort_index().to_dict()
    result['decade_distribution'] = {f"{decade}s": int(count) for decade, count in decades.items()}

    # Month distribution
    month_distribution = dates[valid_mask].dt.month.value_counts().sort_index().to_dict()
    result['month_distribution'] = {str(month): int(count) for month, count in month_distribution.items()}

    # Day of week distribution
    dow_distribution = dates[valid_mask].dt.dayofweek.value_counts().sort_index().to_dict()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    result['day_of_week_distribution'] = {day_names[dow]: int(count) for dow, count in dow_distribution.items()}

    return result


def validate_date_format(date_str: str, format_str: str = '%Y-%m-%d') -> bool:
    """
    Check if a date string matches the specified format.

    Parameters:
    -----------
    date_str : str
        The date string to validate
    format_str : str
        The expected date format

    Returns:
    --------
    bool
        True if the date matches the format, False otherwise
    """
    try:
        datetime.strptime(date_str, format_str)
        return True
    except (ValueError, TypeError):
        return False


def detect_date_anomalies(dates: pd.Series, min_year: int = 1940, max_year: int = 2005) -> Dict[str, List[Any]]:
    """
    Detect anomalies in dates (too old, future dates, invalid formats).

    Parameters:
    -----------
    dates : pd.Series
        Series of dates to analyze
    min_year : int
        Minimum valid year
    max_year : int
        Maximum valid year

    Returns:
    --------
    Dict[str, List[Any]]
        Dictionary with anomaly categories and examples
    """
    # Convert dates to datetime with errors flagged
    date_objects = pd.to_datetime(dates, errors='coerce')

    # Initialize results
    anomalies = {
        'invalid_format': [],  # Invalid format
        'too_old': [],  # Too old (before min_year)
        'future_dates': [],  # Future dates
        'too_young': [],  # Too young (after max_year)
        'negative_years': []  # Negative years
    }

    # Check each date
    for i, date_str in enumerate(dates):
        if pd.isna(date_str):
            continue

        # Check format
        if not validate_date_format(str(date_str)):
            anomalies['invalid_format'].append((i, date_str))
            continue

        # If format is valid but date doesn't convert, look for the reason
        if pd.isna(date_objects[i]):
            if str(date_str).startswith('-'):
                anomalies['negative_years'].append((i, date_str))
            else:
                anomalies['invalid_format'].append((i, date_str))
        else:
            # For convertible dates, check the range
            year = date_objects[i].year

            if year < min_year:
                anomalies['too_old'].append((i, date_str, year))
            elif year > datetime.now().year:
                anomalies['future_dates'].append((i, date_str, year))
            elif year > max_year:
                anomalies['too_young'].append((i, date_str, year))

    return anomalies


def detect_date_changes_within_group(df: pd.DataFrame, group_column: str, date_column: str) -> Dict[str, Any]:
    """
    Detect date changes within groups (e.g., resume_id).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    group_column : str
        The column to group by
    date_column : str
        The date column to analyze

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    # Check if columns exist
    if group_column not in df.columns:
        return {'error': f"Group column {group_column} not found in DataFrame"}

    if date_column not in df.columns:
        return {'error': f"Date column {date_column} not found in DataFrame"}

    # Group by the specified column
    grouped = df.groupby(group_column)

    # Look for groups with varying dates
    results = {
        'groups_with_changes': 0,
        'examples': []
    }

    for group_id, group_df in grouped:
        # Skip groups with missing dates
        dates = group_df[date_column].dropna()
        if len(dates) <= 1:
            continue

        # If there's more than one unique value
        if dates.nunique() > 1:
            results['groups_with_changes'] += 1

            # Add example if we haven't reached the limit
            if len(results['examples']) < 10:
                results['examples'].append({
                    'group_id': group_id,
                    'date_values': dates.unique().tolist()
                })

    return results


def detect_date_inconsistencies_by_uid(df: pd.DataFrame, uid_column: str, date_column: str) -> Dict[str, Any]:
    """
    Detect date inconsistencies by UID (person identifier).

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to analyze
    uid_column : str
        The UID column
    date_column : str
        The date column to analyze

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    # Check if columns exist
    if uid_column not in df.columns:
        return {'error': f"UID column {uid_column} not found in DataFrame"}

    if date_column not in df.columns:
        return {'error': f"Date column {date_column} not found in DataFrame"}

    # Group by UID
    grouped = df.groupby(uid_column)

    # Look for UIDs with varying dates
    results = {
        'uids_with_inconsistencies': 0,
        'examples': []
    }

    for uid, group_df in grouped:
        # Skip groups with missing dates
        dates = group_df[date_column].dropna()
        if len(dates) <= 1:
            continue

        # If there's more than one unique value
        if dates.nunique() > 1:
            results['uids_with_inconsistencies'] += 1

            # Add example if we haven't reached the limit
            if len(results['examples']) < 10:
                results['examples'].append({
                    'uid': uid,
                    'date_values': dates.unique().tolist()
                })

    return results


def analyze_date_field(df: pd.DataFrame,
                       field_name: str,
                       min_year: int = 1940,
                       max_year: int = 2005,
                       id_column: Optional[str] = None,
                       uid_column: Optional[str] = None,
                       **kwargs) -> Dict[str, Any]:
    """
    Comprehensive analysis of a date field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    min_year : int
        Minimum valid year for anomaly detection
    max_year : int
        Maximum valid year for anomaly detection
    id_column : str, optional
        The column to use for group analysis
    uid_column : str, optional
        The column to use for UID analysis
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    logger.info(f"Analyzing date field: {field_name}")

    # Basic validation
    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Get prepared data
    dates, null_count, non_null_count = prepare_date_data(df, field_name)
    total_records = len(df)

    # Calculate valid and invalid counts
    valid_mask = ~dates.isna()
    valid_count = valid_mask.sum()
    invalid_count = non_null_count - valid_count

    # Calculate fill and validity rates
    fill_rate = round((non_null_count / total_records) * 100, 2) if total_records > 0 else 0
    valid_rate = round((valid_count / non_null_count) * 100, 2) if non_null_count > 0 else 0

    # Initialize results with basic stats
    results = {
        'total_records': total_records,
        'null_count': int(null_count),
        'non_null_count': int(non_null_count),
        'valid_count': int(valid_count),
        'invalid_count': int(invalid_count),
        'fill_rate': fill_rate,
        'valid_rate': valid_rate
    }

    # Calculate date range and distributions if we have valid dates
    if valid_count > 0:
        # Get date range
        date_stats = calculate_date_stats(dates)
        results.update(date_stats)

        # Calculate distributions
        distributions = calculate_distributions(dates)
        results.update(distributions)

        # Analyze anomalies
        anomalies = detect_date_anomalies(df[field_name], min_year=min_year, max_year=max_year)
        results['anomalies'] = {k: len(v) for k, v in anomalies.items()}

        # Include examples of anomalies
        for anomaly_type, examples in anomalies.items():
            if examples:
                results[f'{anomaly_type}_examples'] = examples[:10]  # First 10 examples

    # Group analysis if id_column is specified
    if id_column and id_column in df.columns:
        group_changes = detect_date_changes_within_group(df, id_column, field_name)
        results['date_changes_within_group'] = group_changes

    # UID analysis if uid_column is specified
    if uid_column and uid_column in df.columns:
        uid_inconsistencies = detect_date_inconsistencies_by_uid(df, uid_column, field_name)
        results['date_inconsistencies_by_uid'] = uid_inconsistencies

    return results


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for date field analysis.

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
    # Basic resource estimation based on DataFrame size
    row_count = len(df)

    # Memory estimation (rough approximation)
    if field_name in df.columns:
        # Estimate based on field type and non-null values
        non_null_count = df[field_name].notna().sum()
        bytes_per_value = 8  # 8 bytes for datetime64

        # Base memory for analysis
        base_memory_mb = 30

        # Memory for field data
        field_memory_mb = (non_null_count * bytes_per_value) / (1024 * 1024)

        # Memory for intermediate calculations
        calc_memory_mb = field_memory_mb * 2  # Multiplication factor for intermediate calculations

        # Total estimated memory
        estimated_memory_mb = base_memory_mb + field_memory_mb + calc_memory_mb

        # Estimated time (very rough approximation)
        if row_count < 10000:
            estimated_time_seconds = 1
        elif row_count < 100000:
            estimated_time_seconds = 3
        elif row_count < 1000000:
            estimated_time_seconds = 15
        else:
            estimated_time_seconds = 60

        return {
            'estimated_memory_mb': estimated_memory_mb,
            'estimated_time_seconds': estimated_time_seconds,
            'recommended_chunk_size': min(100000, max(10000, row_count // 10)),
            'use_chunks_recommended': row_count > 100000
        }
    else:
        # Field not found, return minimal estimates
        return {
            'estimated_memory_mb': 10,
            'estimated_time_seconds': 1,
            'error': f"Field {field_name} not found in DataFrame"
        }