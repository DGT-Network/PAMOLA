"""
Utility functions for analyzing multi-valued fields (MVF) in the HHR project.

This module provides pure analytical functions for MVF analysis,
separate from operation logic, focusing on parsing, metrics calculation,
pattern extraction, and data preparation.

MVF fields contain multiple values per record, typically stored as:
- String representations of arrays: "['Value1', 'Value2']"
- JSON arrays: ["Value1", "Value2"]
- Comma-separated values: "Value1, Value2"
"""

import ast
import json
import logging
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional, Union

import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def parse_mvf(value: Any,
              format_type: Optional[str] = None,
              separator: str = ',',
              quote_char: str = '"',
              array_markers: Tuple[str, str] = ('[', ']'),
              handle_json: bool = True) -> List[str]:
    """
    Parse a multi-valued field value into a list of individual values.

    Parameters:
    -----------
    value : Any
        The MVF value to parse
    format_type : str, optional
        Format type hint: 'json', 'array_string', 'csv', or None (auto-detect)
    separator : str
        Character used to separate values
    quote_char : str
        Character used for quoting values
    array_markers : Tuple[str, str]
        Start and end markers for array representation
    handle_json : bool
        Whether to attempt parsing as JSON

    Returns:
    --------
    List[str]
        List of individual values
    """
    # Handle None, NaN, and empty values
    if pd.isna(value) or value == '':
        return []

    # Handle non-string values
    if not isinstance(value, str):
        return [str(value)]

    # Remove leading/trailing whitespace
    value = value.strip()

    # Handle empty array representations
    if value == '[]' or value == 'None' or value == 'nan':
        return []

    # Use specified format if provided
    if format_type:
        if format_type == 'json':
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item).strip() for item in parsed]
                elif isinstance(parsed, dict):
                    return [str(key).strip() for key in parsed.keys()]
                else:
                    return [str(parsed)]
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Failed to parse value as JSON despite format hint: {value}")

        elif format_type == 'array_string':
            try:
                if value.startswith('[') and value.endswith(']'):
                    parsed_value = ast.literal_eval(value)
                    if isinstance(parsed_value, list):
                        return [str(item).strip() for item in parsed_value]
            except (SyntaxError, ValueError):
                logger.warning(f"Failed to parse value as array string despite format hint: {value}")

        elif format_type == 'csv':
            return [item.strip() for item in value.split(separator) if item.strip()]

    # Auto-detect format and parse

    # Try parsing as JSON if enabled
    if handle_json and ((value.startswith('{') and value.endswith('}')) or
                        (value.startswith('[') and value.endswith(']'))):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed]
            elif isinstance(parsed, dict):
                return [str(key).strip() for key in parsed.keys()]
            else:
                return [str(parsed)]
        except (json.JSONDecodeError, TypeError):
            # If JSON parsing fails, continue with other methods
            pass

    # Try parsing as Python literal (e.g., "['value1', 'value2']")
    if value.startswith(array_markers[0]) and value.endswith(array_markers[1]):
        try:
            parsed_value = ast.literal_eval(value)
            if isinstance(parsed_value, list):
                return [str(item).strip() for item in parsed_value]
        except (SyntaxError, ValueError):
            # If literal parsing fails, try manual parsing
            inner_content = value[1:-1].strip()
            if not inner_content:
                return []

            # Handle quoted values with separators inside them
            result = []
            in_quotes = False
            current_item = ""

            for char in inner_content:
                if char == quote_char:
                    in_quotes = not in_quotes
                elif char == separator and not in_quotes:
                    result.append(current_item.strip().strip(quote_char))
                    current_item = ""
                else:
                    current_item += char

            # Add the last item
            if current_item:
                result.append(current_item.strip().strip(quote_char))

            return result

    # Handle simple separator-based format
    return [item.strip() for item in value.split(separator) if item.strip()]


def detect_mvf_format(values: List[Any]) -> str:
    """
    Detect the most likely format of MVF values in a sample.

    Parameters:
    -----------
    values : List[Any]
        Sample of MVF values to analyze

    Returns:
    --------
    str
        Detected format: 'json', 'array_string', 'csv', or 'unknown'
    """
    # Count occurrences of each format type
    format_counts = Counter()

    # Sample up to 100 non-null values
    sample_values = [v for v in values if not pd.isna(v)][:100]

    for value in sample_values:
        if not isinstance(value, str):
            format_counts['unknown'] += 1
            continue

        value = value.strip()

        # Check for JSON format
        if (value.startswith('[') and value.endswith(']') and
                '"' in value and ',' in value):
            try:
                json.loads(value)
                format_counts['json'] += 1
                continue
            except (json.JSONDecodeError, TypeError):
                pass

        # Check for array string format
        if (value.startswith('[') and value.endswith(']') and
                "'" in value and ',' in value):
            try:
                ast.literal_eval(value)
                format_counts['array_string'] += 1
                continue
            except (SyntaxError, ValueError):
                pass

        # Check for CSV format
        if ',' in value and not (value.startswith('[') and value.endswith(']')):
            format_counts['csv'] += 1
            continue

        format_counts['unknown'] += 1

    # Return the most common format if it's significant
    if format_counts:
        most_common = format_counts.most_common(1)[0]
        if most_common[1] > len(sample_values) * 0.5:
            return most_common[0]

    return 'unknown'


def standardize_mvf_format(value: Any, target_format: str = 'list') -> Union[List[str], str]:
    """
    Standardize an MVF value to a specified format.

    Parameters:
    -----------
    value : Any
        The MVF value to standardize
    target_format : str
        Target format: 'list', 'json', 'csv', or 'array_string'

    Returns:
    --------
    Union[List[str], str]
        Standardized MVF value
    """
    # Parse the MVF value to get a list of values
    values = parse_mvf(value)

    # Return in the target format
    if target_format == 'list':
        return values
    elif target_format == 'json':
        return json.dumps(values)
    elif target_format == 'csv':
        return ', '.join(values)
    elif target_format == 'array_string':
        formatted_values = []
        for v in values:
            # Properly escape single quotes in the values
            escaped_v = v.replace("'", "\\'")
            formatted_values.append(f"'{escaped_v}'")
        return f"[{', '.join(formatted_values)}]"
    else:
        logger.warning(f"Unknown target format: {target_format}. Returning list.")
        return values


def analyze_mvf_field(df: pd.DataFrame,
                      field_name: str,
                      **kwargs) -> Dict[str, Any]:
    """
    Analyze a multi-valued field in the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    **kwargs : dict
        Additional parameters:
        - top_n: Number of top items to include (default: 20)
        - min_frequency: Minimum frequency for dictionaries (default: 1)
        - parse_args: Dict of arguments to pass to parse_mvf

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis
    """
    logger.info(f"Analyzing MVF field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Extract parameters
    top_n = kwargs.get('top_n', 20)
    min_frequency = kwargs.get('min_frequency', 1)
    parse_args = kwargs.get('parse_args', {})

    try:
        # Get basic statistics
        total_records = len(df)
        null_count = df[field_name].isna().sum()
        non_null_count = total_records - null_count
        null_percentage = round((null_count / total_records) * 100, 2) if total_records > 0 else 0

        # Parse MVF values
        parsed_values = []
        value_counts = []
        combinations = []
        empty_arrays_count = 0
        error_count = 0

        # Process each non-null value
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **parse_args)
                parsed_values.extend(values)
                value_counts.append(len(values))

                if not values:
                    empty_arrays_count += 1

                combinations.append(tuple(sorted(values)))
            except Exception as e:
                error_count += 1
                if error_count <= 10:  # Log only first 10 errors to avoid flood
                    logger.warning(f"Error parsing MVF value '{value}': {str(e)}")
                elif error_count == 11:
                    logger.warning("Too many parsing errors. Further errors will not be logged.")

                # Add empty list for error cases
                value_counts.append(0)
                combinations.append(tuple())

                # Check if error limit reached
                if error_count > 1000:
                    return {
                        'error': f"Too many parsing errors (>1000) in field {field_name}",
                        'total_records': total_records,
                        'error_count': error_count
                    }

        # Calculate statistics
        unique_values_count = len(set(parsed_values)) if parsed_values else 0
        avg_values_per_record = sum(value_counts) / len(value_counts) if value_counts else 0

        # Analyze individual values
        value_counter = Counter(parsed_values)
        values_analysis = {str(k): int(v) for k, v in value_counter.most_common(top_n)}

        # Analyze combinations
        combination_counter = Counter(combinations)
        combinations_analysis = {
            ', '.join(combo) if combo else 'Empty': count
            for combo, count in combination_counter.most_common(top_n)
        }

        # Analyze value counts distribution
        value_counts_counter = Counter(value_counts)
        value_counts_distribution = {str(k): int(v) for k, v in sorted(value_counts_counter.items())}

        # Prepare result stats
        stats = {
            'field_name': field_name,
            'total_records': total_records,
            'null_count': int(null_count),
            'null_percentage': null_percentage,
            'non_null_count': int(non_null_count),
            'empty_arrays_count': int(empty_arrays_count),
            'empty_arrays_percentage': round((empty_arrays_count / non_null_count) * 100, 2)
            if non_null_count > 0 else 0,
            'unique_values': unique_values_count,
            'unique_combinations': len(combination_counter),
            'avg_values_per_record': round(avg_values_per_record, 2),
            'max_values_per_record': max(value_counts) if value_counts else 0,
            'values_analysis': values_analysis,
            'combinations_analysis': combinations_analysis,
            'value_counts_distribution': value_counts_distribution
        }

        # Add error count if any errors occurred
        if error_count > 0:
            stats['error_count'] = error_count
            stats['error_percentage'] = round((error_count / total_records) * 100, 2)

        return stats

    except Exception as e:
        logger.error(f"Error analyzing MVF field {field_name}: {str(e)}", exc_info=True)
        return {
            'error': f"Error analyzing MVF field {field_name}: {str(e)}",
            'field_name': field_name
        }


def create_value_dictionary(df: pd.DataFrame,
                            field_name: str,
                            min_frequency: int = 1,
                            parse_args: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Create a dictionary of values with frequencies for an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    parse_args : Dict[str, Any], optional
        Arguments to pass to parse_mvf

    Returns:
    --------
    pd.DataFrame
        DataFrame with values and frequencies
    """
    logger.info(f"Creating value dictionary for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return pd.DataFrame(columns=['value', 'frequency', 'percentage'])

    parse_args = parse_args or {}

    try:
        # Parse MVF values
        all_values = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **parse_args)
                all_values.extend(values)
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")

        # Count frequencies
        value_counter = Counter(all_values)

        # Filter by minimum frequency
        filtered_counter = {k: v for k, v in value_counter.items() if v >= min_frequency}

        if not filtered_counter:
            return pd.DataFrame(columns=['value', 'frequency', 'percentage'])

        # Create DataFrame
        values_df = pd.DataFrame({
            'value': list(filtered_counter.keys()),
            'frequency': list(filtered_counter.values())
        })

        # Calculate percentages
        total = values_df['frequency'].sum()
        values_df['percentage'] = values_df['frequency'] / total * 100 if total > 0 else 0

        # Sort by frequency in descending order
        return values_df.sort_values('frequency', ascending=False)

    except Exception as e:
        logger.error(f"Error creating value dictionary for {field_name}: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=['value', 'frequency', 'percentage'])


def create_combinations_dictionary(df: pd.DataFrame,
                                   field_name: str,
                                   min_frequency: int = 1,
                                   parse_args: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Create a dictionary of value combinations with frequencies for an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    parse_args : Dict[str, Any], optional
        Arguments to pass to parse_mvf

    Returns:
    --------
    pd.DataFrame
        DataFrame with combinations and frequencies
    """
    logger.info(f"Creating combinations dictionary for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return pd.DataFrame(columns=['combination', 'frequency', 'percentage'])

    parse_args = parse_args or {}

    try:
        # Parse MVF values and create combinations
        combinations = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **parse_args)
                combinations.append(tuple(sorted(values)))
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")

        # Count frequencies
        combination_counter = Counter(combinations)

        # Filter by minimum frequency
        filtered_counter = {k: v for k, v in combination_counter.items() if v >= min_frequency}

        if not filtered_counter:
            return pd.DataFrame(columns=['combination', 'frequency', 'percentage'])

        # Create DataFrame
        combinations_df = pd.DataFrame({
            'combination': [', '.join(combo) if combo else 'Empty' for combo in filtered_counter.keys()],
            'frequency': list(filtered_counter.values())
        })

        # Calculate percentages
        total = combinations_df['frequency'].sum()
        combinations_df['percentage'] = combinations_df['frequency'] / total * 100 if total > 0 else 0

        # Sort by frequency in descending order
        return combinations_df.sort_values('frequency', ascending=False)

    except Exception as e:
        logger.error(f"Error creating combinations dictionary for {field_name}: {str(e)}", exc_info=True)
        return pd.DataFrame(columns=['combination', 'frequency', 'percentage'])


def analyze_value_count_distribution(df: pd.DataFrame,
                                     field_name: str,
                                     parse_args: Dict[str, Any] = None) -> Dict[str, int]:
    """
    Analyze the distribution of value counts per record in an MVF field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    parse_args : Dict[str, Any], optional
        Arguments to pass to parse_mvf

    Returns:
    --------
    Dict[str, int]
        Distribution of value counts
    """
    logger.info(f"Analyzing value count distribution for MVF field: {field_name}")

    if field_name not in df.columns:
        logger.error(f"Field {field_name} not found in DataFrame")
        return {}

    parse_args = parse_args or {}

    try:
        # Count values per record
        value_counts = []
        for value in df[field_name].dropna():
            try:
                values = parse_mvf(value, **parse_args)
                value_counts.append(len(values))
            except Exception as e:
                logger.warning(f"Error parsing MVF value '{value}': {str(e)}")
                value_counts.append(0)  # Count as 0 for error cases

        # Count frequencies
        counts_counter = Counter(value_counts)

        # Sort by count number
        return {str(k): int(v) for k, v in sorted(counts_counter.items())}

    except Exception as e:
        logger.error(f"Error analyzing value count distribution for {field_name}: {str(e)}", exc_info=True)
        return {}


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for analyzing an MVF field.

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

    # Estimate basic metrics
    total_records = len(df)
    null_count = df[field_name].isna().sum()
    non_null_count = total_records - null_count

    # Sample a few values to estimate complexity
    sample_size = min(100, non_null_count)
    sample_values = df[field_name].dropna().sample(sample_size) if sample_size > 0 else []

    # Estimate average values per record
    total_values = 0
    complex_values = 0

    for value in sample_values:
        try:
            values = parse_mvf(value)
            total_values += len(values)
            if len(values) > 5:
                complex_values += 1
        except Exception:
            complex_values += 1

    avg_values_per_record = total_values / len(sample_values) if sample_values.size > 0 else 0
    complex_percentage = (complex_values / len(sample_values) * 100) if sample_values.size > 0 else 0

    # Detect format
    detected_format = detect_mvf_format(sample_values)

    # Calculate approximate memory requirement
    memory_per_value = 200  # bytes per unique value with overhead
    estimated_memory = (non_null_count * avg_values_per_record * memory_per_value) / (1024 * 1024)  # in MB

    # Estimate processing time based on row count and complexity
    base_time = 0.1  # seconds
    per_row_time = 0.0001  # seconds per row
    per_value_time = 0.00005  # seconds per value

    estimated_time = base_time + (non_null_count * per_row_time) + (
                non_null_count * avg_values_per_record * per_value_time)

    # Scale up for complex values
    if complex_percentage > 20:
        estimated_time *= 1.5
        estimated_memory *= 1.2

    return {
        'field_name': field_name,
        'total_records': total_records,
        'non_null_count': int(non_null_count),
        'detected_format': detected_format,
        'estimated_avg_values_per_record': round(avg_values_per_record, 2),
        'complex_values_percentage': round(complex_percentage, 2),
        'estimated_memory_mb': round(estimated_memory, 2),
        'estimated_time_seconds': round(estimated_time, 2),
        'large_dataset': total_records > 1000000,
        'dask_recommended': total_records > 1000000 and estimated_memory > 500
    }