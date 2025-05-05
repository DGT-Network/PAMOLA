"""
Utility functions for analyzing categorical fields in the HHR project.

This module provides pure analytical functions for categorical data analysis,
separate from operation logic, focusing on metrics calculation, pattern extraction,
and data preparation.
"""

import logging
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# Configure logger
logger = logging.getLogger(__name__)


def analyze_categorical_field(
        df: pd.DataFrame,
        field_name: str,
        top_n: int = 15,
        min_frequency: int = 1,
        **kwargs
) -> Dict[str, Any]:
    """
    Analyze a categorical field in the given DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data to analyze
    field_name : str
        The name of the field to analyze
    top_n : int
        Number of top values to include in the results
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        The results of the analysis containing statistics and distributions
    """
    logger.info(f"Analyzing categorical field: {field_name}")

    if field_name not in df.columns:
        return {'error': f"Field {field_name} not found in DataFrame"}

    # Get basic statistics
    total_records = len(df)
    non_null_values = df[field_name].count()
    null_values = total_records - non_null_values
    null_percent = round(null_values / total_records * 100, 2) if total_records > 0 else 0
    unique_values = df[field_name].nunique()

    # Get value counts for non-null values
    value_counts = df[field_name].value_counts()

    # Get top values
    top_values_series = value_counts.head(top_n)
    top_values = {str(k): int(v) for k, v in top_values_series.items()}

    # Calculate distribution metrics
    percent_top = sum(top_values_series) / non_null_values * 100 if non_null_values > 0 else 0

    # Calculate diversity metrics
    entropy = 0
    cardinality_ratio = 0

    if non_null_values > 0:
        # Calculate entropy (measure of diversity)
        probabilities = value_counts / non_null_values
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Calculate cardinality ratio (unique values / non-null values)
        cardinality_ratio = unique_values / non_null_values

    # Create value frequency dictionary (for all values meeting minimum frequency)
    value_dict = create_value_dictionary(
        df=df,
        field_name=field_name,
        min_frequency=min_frequency
    )

    # Create result stats
    stats = {
        'field_name': field_name,
        'total_records': total_records,
        'non_null_values': int(non_null_values),
        'null_values': int(null_values),
        'null_percent': null_percent,
        'unique_values': int(unique_values),
        'top_values': top_values,
        'percent_covered_by_top': round(percent_top, 2),
        'entropy': float(entropy),
        'cardinality_ratio': float(cardinality_ratio),
        'value_dictionary': value_dict
    }

    # Add additional analysis based on kwargs
    if kwargs.get('analyze_distribution', True):
        # Analyze distribution characteristics
        distribution_analysis = analyze_distribution_characteristics(
            value_counts=value_counts,
            non_null_values=non_null_values,
            unique_values=unique_values
        )
        stats.update(distribution_analysis)

    # Check for potential anomalies
    if kwargs.get('detect_anomalies', True):
        anomalies = detect_anomalies(
            df=df,
            field_name=field_name,
            value_counts=value_counts,
            min_frequency=kwargs.get('anomaly_threshold', 1)
        )
        if anomalies:
            stats['anomalies'] = anomalies

    return stats


def create_value_dictionary(
        df: pd.DataFrame,
        field_name: str,
        min_frequency: int = 1
) -> Dict[str, Any]:
    """
    Create a frequency dictionary for a categorical field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    min_frequency : int
        Minimum frequency for inclusion in the dictionary

    Returns:
    --------
    Dict[str, Any]
        Dictionary with frequency data and metadata
    """
    logger.info(f"Creating value dictionary for field: {field_name}")

    try:
        # Get frequency distribution
        value_counts = df[field_name].value_counts().reset_index()
        value_counts.columns = [field_name, 'frequency']

        # Filter by minimum frequency
        value_counts = value_counts[value_counts['frequency'] >= min_frequency]

        # Calculate percentages
        total = value_counts['frequency'].sum()
        value_counts['percent'] = (value_counts['frequency'] / total * 100).round(2)

        # Convert to dictionary format for returning
        result = {
            'field_name': field_name,
            'total_unique_values': len(value_counts),
            'min_frequency': int(value_counts['frequency'].min()) if not value_counts.empty else 0,
            'max_frequency': int(value_counts['frequency'].max()) if not value_counts.empty else 0,
            'dictionary_data': value_counts.to_dict(orient='records')
        }

        return result

    except Exception as e:
        logger.error(f"Error creating dictionary for {field_name}: {e}", exc_info=True)
        return {
            'error': str(e),
            'field_name': field_name,
            'total_unique_values': 0,
            'dictionary_data': []
        }


def analyze_distribution_characteristics(
        value_counts: pd.Series,
        non_null_values: int,
        unique_values: int
) -> Dict[str, Any]:
    """
    Analyze the characteristics of a categorical distribution.

    Parameters:
    -----------
    value_counts : pd.Series
        Series of value counts
    non_null_values : int
        Number of non-null values
    unique_values : int
        Number of unique values

    Returns:
    --------
    Dict[str, Any]
        Dictionary with distribution characteristics
    """
    if value_counts.empty or non_null_values == 0:
        return {
            'distribution_type': 'empty',
            'skewness': 0,
            'concentration': 0
        }

    # Calculate skewness of distribution
    # Higher values indicate more concentration in a few categories
    if len(value_counts) >= 2:
        # Normalize value counts to get probabilities
        probs = value_counts / non_null_values

        # Calculate variance
        mean_prob = 1 / len(probs)
        variance = ((probs - mean_prob) ** 2).sum() / len(probs)

        # Calculate skewness
        skewness = ((probs - mean_prob) ** 3).sum() / (len(probs) * variance ** 1.5) if variance > 0 else 0
    else:
        skewness = 0

    # Calculate concentration ratio (Gini-like)
    # 0 means perfect equality, 1 means total concentration
    concentration = 0
    if len(value_counts) >= 2:
        top_value = value_counts.iloc[0]
        total_values = value_counts.sum()
        concentration = (top_value - (total_values / len(value_counts))) / top_value

    # Determine distribution type
    distribution_type = "unknown"
    if unique_values == 1:
        distribution_type = "single_value"
    elif concentration > 0.9:
        distribution_type = "highly_concentrated"
    elif concentration > 0.7:
        distribution_type = "concentrated"
    elif concentration > 0.4:
        distribution_type = "moderately_distributed"
    else:
        distribution_type = "well_distributed"

    return {
        'distribution_type': distribution_type,
        'skewness': float(skewness),
        'concentration': float(concentration)
    }


def detect_anomalies(
        df: pd.DataFrame,
        field_name: str,
        value_counts: pd.Series,
        min_frequency: int = 1
) -> Dict[str, Any]:
    """
    Detect potential anomalies in categorical field values.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    field_name : str
        The name of the field
    value_counts : pd.Series
        Series of value counts
    min_frequency : int
        Minimum frequency for anomaly detection

    Returns:
    --------
    Dict[str, Any]
        Dictionary with detected anomalies
    """
    anomalies = {}

    # Detect single-character values that might be errors
    single_char_values = [val for val in value_counts.index if isinstance(val, str) and len(val.strip()) == 1]
    if single_char_values:
        single_char_counts = {str(val): int(value_counts[val]) for val in single_char_values
                              if value_counts[val] <= min_frequency}
        if single_char_counts:
            anomalies['single_char_values'] = single_char_counts

    # Detect numeric-like strings that should be numbers
    numeric_like_strings = [val for val in value_counts.index
                            if isinstance(val, str) and val.strip().replace('.', '').isdigit()]
    if numeric_like_strings:
        numeric_strings_counts = {str(val): int(value_counts[val]) for val in numeric_like_strings}
        if numeric_strings_counts:
            anomalies['numeric_like_strings'] = numeric_strings_counts

    # Detect potential typos (rare values similar to common values)
    common_values = set([str(val) for val in value_counts.index
                         if value_counts[val] > min_frequency and isinstance(val, str)])
    rare_values = set([str(val) for val in value_counts.index
                       if value_counts[val] <= min_frequency and isinstance(val, str)])

    potential_typos = {}
    for rare in rare_values:
        for common in common_values:
            # Simple string similarity check (can be improved)
            if (rare != common and
                    (rare in common or common in rare or
                     levenshtein_distance(rare, common) <= 2)):
                potential_typos[rare] = {
                    'count': int(value_counts[rare]),
                    'similar_to': common,
                    'similar_count': int(value_counts[common])
                }

    if potential_typos:
        anomalies['potential_typos'] = potential_typos

    return anomalies


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate the Levenshtein distance between two strings.

    Parameters:
    -----------
    s1 : str
        First string
    s2 : str
        Second string

    Returns:
    --------
    int
        The Levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def estimate_resources(df: pd.DataFrame, field_name: str) -> Dict[str, Any]:
    """
    Estimate resources needed for analyzing a categorical field.

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

    # Count unique values to estimate dictionary size
    unique_count = df[field_name].nunique()

    # Calculate approximate memory requirement based on unique values
    # This is a rough estimate - adjust coefficients based on profiling
    memory_per_value = 200  # bytes per unique value with overhead
    estimated_memory = (unique_count * memory_per_value) / (1024 * 1024)  # in MB

    # Estimate processing time based on row count and unique values
    # Again, these are rough estimates
    base_time = 0.1  # seconds
    per_row_time = 0.00001  # seconds per row
    per_unique_time = 0.0001  # seconds per unique value

    estimated_time = base_time + (len(df) * per_row_time) + (unique_count * per_unique_time)

    # Determine if distribution is complex based on unique values
    complex_distribution = unique_count > 1000

    return {
        'estimated_memory_mb': round(estimated_memory, 2),
        'estimated_time_seconds': round(estimated_time, 2),
        'complex_distribution': complex_distribution,
        'unique_value_count': unique_count,
        'total_records': len(df)
    }


def analyze_multiple_categorical_fields(
        df: pd.DataFrame,
        fields: List[str],
        top_n: int = 15,
        min_frequency: int = 1,
        **kwargs
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze multiple categorical fields in a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    fields : List[str]
        List of field names to analyze
    top_n : int
        Number of top values to include in the results
    min_frequency : int
        Minimum frequency for inclusion in the dictionary
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Dict[str, Any]]
        Dictionary mapping field names to their analysis results
    """
    results = {}

    for field in fields:
        try:
            logger.info(f"Analyzing field: {field}")
            field_result = analyze_categorical_field(
                df=df,
                field_name=field,
                top_n=top_n,
                min_frequency=min_frequency,
                **kwargs
            )
            results[field] = field_result
        except Exception as e:
            logger.error(f"Error analyzing field {field}: {e}", exc_info=True)
            results[field] = {'error': str(e)}

    return results