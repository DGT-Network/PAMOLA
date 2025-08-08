"""
AMOLA.CORE - Privacy-Preserving AI Data Processors
------------------------------------------------------------
Module:        Privacy-Specific Data Processing Utilities
Package:       pamola_core.anonymization.commons
Version:       1.1.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
   This module provides privacy-specific data processing utilities for anonymization
   operations. It focuses exclusively on privacy-aware functionality that extends
   the general-purpose utilities from pamola_core.utils.ops modules.

Key Features:
   - Privacy-aware null value handling strategies
   - Risk-based record filtering using k-anonymity scores
   - Vulnerable record handling with various strategies
   - Integration with profiling results and risk assessments
   - Factory functions for creating risk-based processors
   - Adaptive privacy level processors

Framework:
   This module acts as a thin layer over existing framework utilities:
   - Uses op_data_processing.py for general data operations
   - Uses op_field_utils.py for field manipulations
   - Adds privacy-specific logic on top of these utilities

Changelog:
   1.1.0 - Added factory functions and privacy level processors
         - create_risk_based_processor for strategy-based processing
         - create_privacy_level_processor for adaptive anonymization
         - Added constants for common thresholds and strategies
   1.0.0 - Initial implementation with core privacy functions
         - process_nulls for privacy-aware null handling
         - filter_records_conditionally for risk-based filtering
         - handle_vulnerable_records for high-risk record processing

Dependencies:
   - pandas - DataFrame operations
   - numpy - Numeric computations
   - logging - Error and progress reporting
   - typing - Type hints
   - pamola_core.utils.ops.op_data_processing - General data utilities
   - pamola_core.utils.ops.op_field_utils - Field manipulation utilities

TODO:
   - Add support for more sophisticated risk-based strategies
   - Implement validation against profiling results
   - Add adaptive processing based on privacy levels
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable

import numpy as np
import pandas as pd

# Import framework utilities
from pamola_core.utils.ops.op_field_utils import apply_condition_operator

# Configure module logger
logger = logging.getLogger(__name__)

# =============================================================================
# Constants and Configuration
# =============================================================================

# Default thresholds
DEFAULT_K_THRESHOLD = 5
DEFAULT_SUPPRESSION_WARNING = 0.2
DEFAULT_COVERAGE_TARGET = 0.95

# Risk levels for k-anonymity
RISK_LEVELS = {
    "VERY_HIGH": (0, 2),
    "HIGH": (2, 5),
    "MEDIUM": (5, 10),
    "LOW": (10, float('inf'))
}

# Null handling strategies
NULL_STRATEGIES = ["PRESERVE", "EXCLUDE", "ANONYMIZE", "ERROR"]

# Vulnerable record handling strategies
VULNERABLE_STRATEGIES = ["suppress", "remove", "mean", "mode", "custom"]

# Privacy levels for adaptive processing
PRIVACY_LEVELS = {
    "LOW": {
        "k_threshold": 2,
        "suppression_limit": 0.3,
        "generalization_level": 0.3
    },
    "MEDIUM": {
        "k_threshold": 5,
        "suppression_limit": 0.2,
        "generalization_level": 0.5
    },
    "HIGH": {
        "k_threshold": 10,
        "suppression_limit": 0.1,
        "generalization_level": 0.7
    },
    "VERY_HIGH": {
        "k_threshold": 20,
        "suppression_limit": 0.05,
        "generalization_level": 0.9
    }
}


# =============================================================================
# Privacy-Aware Null Processing
# =============================================================================

def process_nulls(series: pd.Series,
                  strategy: str = "PRESERVE",
                  anonymize_value: str = "SUPPRESSED") -> pd.Series:
    """
    Process null values with privacy-aware strategies.

    This function extends the general null processing with privacy-specific
    strategies, particularly the ANONYMIZE option that replaces nulls with
    a privacy-preserving placeholder.

    Parameters:
    -----------
    series : pd.Series
        The series containing null values to process
    strategy : str, optional
        Null handling strategy (default: "PRESERVE")
        - "PRESERVE": Keep null values as is
        - "EXCLUDE": Remove records with null values
        - "ANONYMIZE": Replace nulls with anonymize_value
    anonymize_value : str, optional
        Value to use when strategy is "ANONYMIZE" (default: "SUPPRESSED")

    Returns:
    --------
    pd.Series
        Series with processed null values. Note that when using "ANONYMIZE"
        with numeric series, the return type will be 'object' to accommodate
        the string replacement value.

    Raises:
    -------
    TypeError
        If series is not a pandas Series
    ValueError
        If strategy is not recognized

    Notes:
    ------
    - The "EXCLUDE" strategy returns a series with null values removed.
      For DataFrame-level filtering, handle this at the caller level.
    - When anonymizing numeric series with string values, the series
      dtype will change to 'object'.

    Examples:
    ---------
    >>> s = pd.Series([1, 2, None, 4, None])
    >>> process_nulls(s, strategy="ANONYMIZE")
    0            1
    1            2
    2    SUPPRESSED
    3            4
    4    SUPPRESSED
    dtype: object

    >>> process_nulls(s, strategy="PRESERVE")
    0    1.0
    1    2.0
    2    NaN
    3    4.0
    4    NaN
    dtype: float64
    """
    # Type validation
    if not isinstance(series, pd.Series):
        raise TypeError(f"Argument 'series' must be a pandas.Series, got {type(series).__name__}")

    if strategy not in NULL_STRATEGIES:
        raise ValueError(f"Invalid null strategy: {strategy}. Must be one of {NULL_STRATEGIES}")

    if strategy == "PRESERVE":
        # Return series as-is, no modification needed
        return series.copy()

    elif strategy == "EXCLUDE":
        # Remove null values
        # Note: This changes the series length, which may not align with
        # the original DataFrame. Consider using at DataFrame level instead.
        return series.dropna()

    elif strategy == "ANONYMIZE":
        # Privacy-specific: replace nulls with anonymized value
        if series.isna().sum() == 0:
            # No nulls to anonymize
            return series.copy()

        if pd.api.types.is_numeric_dtype(series) and not isinstance(anonymize_value, (int, float)):
            # Need to convert to object type to accommodate string anonymize_value
            # Explicitly create a copy to avoid any potential side effects
            result = series.astype('object').copy()
            result.loc[series.isna()] = anonymize_value
            return result
        else:
            # Direct replacement (categorical, string, or numeric with numeric anonymize_value)
            result = series.copy()
            result = result.fillna(anonymize_value)
            return result

    else:
        # This should never be reached due to earlier validation
        # but included for static analysis and defensive programming
        raise RuntimeError(f"Unexpected strategy branch: {strategy}")


# =============================================================================
# Risk-Based Record Filtering
# =============================================================================

def filter_records_conditionally(df: pd.DataFrame,
                                 risk_field: Optional[str] = None,
                                 risk_threshold: float = DEFAULT_K_THRESHOLD,
                                 operator: str = "ge",
                                 condition_field: Optional[str] = None,
                                 condition_values: Optional[List] = None,
                                 condition_operator: str = "in") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter DataFrame records based on risk scores and optional conditions.

    This function creates a mask for records that should be processed based on
    k-anonymity risk scores (from profiling) and optional additional conditions.
    It's designed to work with profiling results where a risk score field
    indicates the privacy risk level of each record.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to filter
    risk_field : Optional[str]
        Name of the field containing risk scores (e.g., k-anonymity values)
        If None, all records are considered for processing
    risk_threshold : float, optional
        Threshold value for risk filtering (default: 5.0)
        Records are selected based on comparison with this threshold
    operator : str, optional
        Operator for risk comparison (default: "ge")
        - "ge": risk >= threshold (process high-risk records)
        - "lt": risk < threshold (process low-risk records)
    condition_field : Optional[str]
        Additional field for conditional filtering
    condition_values : Optional[List]
        Values for additional condition
    condition_operator : str, optional
        Operator for additional condition (default: "in")

    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        (Filtered DataFrame containing only records to process,
         Boolean mask indicating which records were selected)

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'name': ['Alice', 'Bob', 'Charlie', 'David'],
    ...     'k_score': [2, 10, 3, 15],
    ...     'dept': ['IT', 'HR', 'IT', 'HR']
    ... })
    >>> filtered_df, mask = filter_records_conditionally(
    ...     df, risk_field='k_score', risk_threshold=5, operator='lt'
    ... )
    >>> print(filtered_df)
         name  k_score dept
    0   Alice        2   IT
    2  Charlie        3   IT
    """
    # Initialize with all records selected
    mask = pd.Series(True, index=df.index)

    # Apply risk-based filtering if risk field is specified
    if risk_field is not None and risk_field in df.columns:
        if operator == "ge":
            risk_mask = df[risk_field] >= risk_threshold
        elif operator == "lt":
            risk_mask = df[risk_field] < risk_threshold
        elif operator == "gt":
            risk_mask = df[risk_field] > risk_threshold
        elif operator == "le":
            risk_mask = df[risk_field] <= risk_threshold
        else:
            raise ValueError(f"Invalid operator: {operator}. Must be one of ['ge', 'lt', 'gt', 'le']")

        mask = mask & risk_mask

        logger.info(f"Risk-based filtering: {risk_mask.sum()}/{len(df)} records have {operator} {risk_threshold}")

    # Apply additional conditional filtering if specified
    if condition_field is not None and condition_field in df.columns:
        # Use framework utility for condition application
        condition_mask = apply_condition_operator(
            df[condition_field],
            condition_values,
            condition_operator
        )
        mask = mask & condition_mask

        logger.info(f"Additional filtering: {condition_mask.sum()}/{len(df)} records match condition")

    # Filter the DataFrame
    filtered_df = df[mask].copy()

    logger.info(f"Total filtering result: {len(filtered_df)}/{len(df)} records selected for processing")

    return filtered_df, mask


# =============================================================================
# Vulnerable Record Handling
# =============================================================================

def handle_vulnerable_records(df: pd.DataFrame,
                              field_name: str,
                              vulnerability_mask: pd.Series,
                              strategy: str = "suppress",
                              replacement_value: Optional[Any] = None) -> pd.DataFrame:
    """
    Handle vulnerable records identified by risk assessment.

    This function processes records identified as high-risk (vulnerable) by
    applying various privacy-preserving strategies. It's typically used after
    filter_records_conditionally identifies which records need special handling.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    field_name : str
        Name of the field to process for vulnerable records
    vulnerability_mask : pd.Series
        Boolean mask indicating vulnerable records (True = vulnerable)
    strategy : str, optional
        Strategy for handling vulnerable records (default: "suppress")
        - "suppress": Replace with "SUPPRESSED" or null
        - "remove": Remove the vulnerable records entirely
        - "mean": Replace with mean value (numeric fields only)
        - "mode": Replace with mode value (categorical fields)
        - "custom": Replace with replacement_value
    replacement_value : Optional[Any]
        Custom value to use when strategy is "custom"

    Returns:
    --------
    pd.DataFrame
        DataFrame with vulnerable records handled according to strategy

    Raises:
    -------
    ValueError
        If strategy is not recognized or incompatible with field type

    Examples:
    ---------
    >>> df = pd.DataFrame({
    ...     'salary': [50000, 60000, 55000, 65000],
    ...     'risk': [10, 2, 3, 1]
    ... })
    >>> mask = df['risk'] < 5  # Vulnerable records
    >>> result = handle_vulnerable_records(
    ...     df, 'salary', mask, strategy='mean'
    ... )
    >>> print(result['salary'])
    0    50000.0
    1    60000.0  # Replaced with mean
    2    60000.0  # Replaced with mean
    3    60000.0  # Replaced with mean
    """
    if strategy not in VULNERABLE_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {VULNERABLE_STRATEGIES}")

    if field_name not in df.columns:
        raise ValueError(f"Field '{field_name}' not found in DataFrame")

    # Count vulnerable records
    vulnerable_count = vulnerability_mask.sum()
    if vulnerable_count == 0:
        logger.info("No vulnerable records found, returning original DataFrame")
        return df.copy()

    logger.info(f"Processing {vulnerable_count} vulnerable records using strategy: {strategy}")

    # Create a copy to avoid modifying original
    result_df = df.copy()

    if strategy == "remove":
        # Remove vulnerable records
        result_df = result_df[~vulnerability_mask].copy()
        logger.info(f"Removed {vulnerable_count} vulnerable records")

    elif strategy == "suppress":
        # Replace with suppression marker
        if pd.api.types.is_numeric_dtype(df[field_name]):
            # For numeric fields, use NaN or convert to object type
            result_df.loc[vulnerability_mask, field_name] = np.nan
        else:
            result_df.loc[vulnerability_mask, field_name] = "SUPPRESSED"

    elif strategy == "mean":
        # Replace with mean (numeric fields only)
        if not pd.api.types.is_numeric_dtype(df[field_name]):
            raise ValueError(f"Strategy 'mean' requires numeric field, but '{field_name}' is not numeric")

        # Calculate mean from non-vulnerable records
        safe_mean = df.loc[~vulnerability_mask, field_name].mean()
        if pd.isna(safe_mean):
            # Fallback to overall mean if no safe records
            safe_mean = df[field_name].mean()

        result_df.loc[vulnerability_mask, field_name] = safe_mean
        logger.info(f"Replaced vulnerable records with mean value: {safe_mean:.2f}")

    elif strategy == "mode":
        # Replace with mode (most common value)
        # Calculate mode from non-vulnerable records
        safe_values = df.loc[~vulnerability_mask, field_name]
        if len(safe_values) > 0:
            mode_value = safe_values.mode()
            if len(mode_value) > 0:
                mode_value = mode_value.iloc[0]
            else:
                # Fallback to overall mode
                mode_value = df[field_name].mode().iloc[0]
        else:
            # All records are vulnerable, use overall mode
            mode_value = df[field_name].mode().iloc[0]

        result_df.loc[vulnerability_mask, field_name] = mode_value
        logger.info(f"Replaced vulnerable records with mode value: {mode_value}")

    elif strategy == "custom":
        # Replace with custom value
        if replacement_value is None:
            raise ValueError("replacement_value must be provided when using 'custom' strategy")

        result_df.loc[vulnerability_mask, field_name] = replacement_value
        logger.info(f"Replaced vulnerable records with custom value: {replacement_value}")

    return result_df


# =============================================================================
# Factory Functions for Risk-Based Processing
# =============================================================================

def create_risk_based_processor(strategy: str = "adaptive",
                                risk_threshold: float = DEFAULT_K_THRESHOLD) -> Callable:
    """
    Factory for creating risk-based processing functions.

    Creates a callable processor that handles vulnerable records based on
    the specified strategy. The returned function can be used in anonymization
    operations to consistently apply the same risk-based processing.

    Parameters:
    -----------
    strategy : str, optional
        Risk handling strategy (default: "adaptive")
        - "conservative": Suppress all vulnerable records
        - "adaptive": Replace with statistical values (mean/mode)
        - "aggressive": Minimal changes, use custom markers
        - "remove": Remove vulnerable records entirely
    risk_threshold : float, optional
        K-anonymity threshold for identifying vulnerable records

    Returns:
    --------
    Callable
        A function that takes (df, field_name, vulnerability_mask) and
        returns processed DataFrame

    Examples:
    ---------
    #>>> vulnerability_mask = (df["salary"] < 100)
    #>>> processed_df = processor(df, "salary", vulnerability_mask)

    """

    def conservative_processor(df: pd.DataFrame, field_name: str,
                               vulnerability_mask: pd.Series) -> pd.DataFrame:
        """Conservative: Suppress all vulnerable records"""
        return handle_vulnerable_records(
            df, field_name, vulnerability_mask,
            strategy="suppress"
        )

    def adaptive_processor(df: pd.DataFrame, field_name: str,
                           vulnerability_mask: pd.Series) -> pd.DataFrame:
        """Adaptive: Use statistical replacement"""
        # Check field type to decide between mean and mode
        if pd.api.types.is_numeric_dtype(df[field_name]):
            return handle_vulnerable_records(
                df, field_name, vulnerability_mask,
                strategy="mean"
            )
        else:
            return handle_vulnerable_records(
                df, field_name, vulnerability_mask,
                strategy="mode"
            )

    def aggressive_processor(df: pd.DataFrame, field_name: str,
                             vulnerability_mask: pd.Series) -> pd.DataFrame:
        """Aggressive: Minimal changes with markers"""
        return handle_vulnerable_records(
            df, field_name, vulnerability_mask,
            strategy="custom",
            replacement_value="*"
        )

    def remove_processor(df: pd.DataFrame, field_name: str,
                         vulnerability_mask: pd.Series) -> pd.DataFrame:
        """Remove: Delete vulnerable records"""
        return handle_vulnerable_records(
            df, field_name, vulnerability_mask,
            strategy="remove"
        )

    # Strategy mapping
    strategies = {
        "conservative": conservative_processor,
        "adaptive": adaptive_processor,
        "aggressive": aggressive_processor,
        "remove": remove_processor
    }

    # Return selected strategy or default to adaptive
    selected_processor = strategies.get(strategy, adaptive_processor)

    # Log the strategy selection
    logger.info(f"Created risk-based processor with strategy: {strategy}")

    return selected_processor


def create_privacy_level_processor(privacy_level: str = "MEDIUM") -> Dict[str, Any]:
    """
    Create a configuration for processing based on privacy level.

    Returns a dictionary of processing parameters optimized for the
    specified privacy level. These parameters can be used across
    different anonymization operations to ensure consistent privacy
    protection.

    Parameters:
    -----------
    privacy_level : str, optional
        Target privacy level (default: "MEDIUM")
        - "LOW": Minimal privacy, maximum utility
        - "MEDIUM": Balanced privacy and utility
        - "HIGH": Strong privacy, reduced utility
        - "VERY_HIGH": Maximum privacy, minimal utility

    Returns:
    --------
    Dict[str, Any]
        Configuration parameters including:
        - k_threshold: Minimum k-anonymity value
        - suppression_limit: Maximum suppression rate
        - generalization_level: Target generalization level
        - risk_processor: Callable for handling vulnerable records
        - null_strategy: Strategy for null handling

    Examples:
    ---------
    >>> privacy_cfg = create_privacy_level_processor("HIGH")
    >>> k_threshold = privacy_cfg["k_threshold"]  # 10
    >>> processor = privacy_cfg["risk_processor"]
    >>> # Suppose you have a DataFrame 'my_df' and a boolean Series 'my_mask':
    >>> # processed_df = processor(my_df, "salary", my_mask)
    """
    if privacy_level not in PRIVACY_LEVELS:
        raise ValueError(
            f"Invalid privacy level: {privacy_level}. "
            f"Must be one of {list(PRIVACY_LEVELS.keys())}"
        )

    # Get base configuration
    cfg = PRIVACY_LEVELS[privacy_level].copy()

    # Add appropriate risk processor based on privacy level
    if privacy_level == "LOW":
        cfg["risk_processor"] = create_risk_based_processor("aggressive") # type: ignore
        cfg["null_strategy"] = "PRESERVE" # type: ignore
        cfg["vulnerable_strategy"] = "aggressive" # type: ignore

    elif privacy_level == "MEDIUM":
        cfg["risk_processor"] = create_risk_based_processor("adaptive") # type: ignore
        cfg["null_strategy"] = "PRESERVE" # type: ignore
        cfg["vulnerable_strategy"] = "adaptive" # type: ignore

    elif privacy_level == "HIGH":
        cfg["risk_processor"] = create_risk_based_processor("conservative") # type: ignore
        cfg["null_strategy"] = "ANONYMIZE" # type: ignore
        cfg["vulnerable_strategy"] = "conservative" # type: ignore

    else:  # VERY_HIGH
        cfg["risk_processor"] = create_risk_based_processor("remove") # type: ignore
        cfg["null_strategy"] = "ANONYMIZE" # type: ignore
        cfg["vulnerable_strategy"] = "remove" # type: ignore

    # Add risk level classification and description
    cfg["risk_levels"] = RISK_LEVELS # type: ignore
    cfg["description"] = f"{privacy_level} privacy level configuration" # type: ignore

    logger.info(f"Created privacy level processor for {privacy_level} privacy")

    return cfg



def apply_adaptive_anonymization(
    df: pd.DataFrame,
    field_name: str,
    risk_scores: pd.Series,
    privacy_level: str = "MEDIUM"
) -> pd.DataFrame:
    """
    Apply adaptive anonymization based on risk scores and privacy level.

    This function applies different anonymization strategies to risk groups in the data,
    providing strong protection for high-risk records and preserving utility for low-risk records.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to anonymize.
    field_name : str
        Column name to anonymize.
    risk_scores : pd.Series
        Risk scores (e.g., k-anonymity) for each record.
    privacy_level : str, optional
        Overall privacy level ("LOW", "MEDIUM", "HIGH", "VERY_HIGH").

    Returns
    -------
    pd.DataFrame
        DataFrame with adaptive anonymization applied.

    Examples
    --------
    >>> import pandas as pd
    >>> test_df = pd.DataFrame({"salary": [100, 200, 300, 400, 500]})
    >>> test_risk_scores = pd.Series([1, 3, 10, 50, 100])
    >>> anonymized = apply_adaptive_anonymization(
    ...     test_df, "salary", test_risk_scores, "HIGH"
    ... )
    """
    # Get privacy config
    config = create_privacy_level_processor(privacy_level)
    result_df = df.copy()

    # Assign each record a risk level
    risk_classification = pd.Series("LOW", index=df.index, dtype="object")
    for risk_level, (min_k, max_k) in RISK_LEVELS.items():
        level_mask: pd.Series = (risk_scores >= min_k) & (risk_scores < max_k) # type: ignore
        risk_classification.loc[level_mask] = risk_level

    # Process each risk group using an explicit Series mask
    for risk_level in ["VERY_HIGH", "HIGH", "MEDIUM", "LOW"]:
        group_mask: pd.Series = (risk_classification == risk_level) # type: ignore
        if group_mask.sum() == 0:
            continue

        logger.info(f"Processing {group_mask.sum()} {risk_level} risk records")

        if risk_level == "VERY_HIGH":
            # Conservative suppression for very high risk
            result_df = handle_vulnerable_records(
                result_df, field_name, group_mask,
                strategy="suppress"
            )
        elif risk_level == "HIGH":
            processor = config["risk_processor"]
            result_df = processor(result_df, field_name, group_mask)
        else:
            # For MEDIUM and LOW risk, apply lighter anonymization if privacy_level is strict
            if privacy_level in ["HIGH", "VERY_HIGH"]:
                result_df = handle_vulnerable_records(
                    result_df, field_name, group_mask,
                    strategy="custom",
                    replacement_value="<ANONYMIZED>"
                )

    return result_df





# =============================================================================
# Utility Functions
# =============================================================================

def get_risk_statistics(df: pd.DataFrame, risk_field: str,
                        thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calculate statistics for risk values in a DataFrame.

    Args:
        df: DataFrame containing risk values
        risk_field: Name of the field containing risk values
        thresholds: List of threshold values for distribution analysis
                   If None, uses standard risk level thresholds

    Returns:
        Dictionary containing statistics and distribution data
    """
    # Use standard thresholds if none provided
    if thresholds is None:
        thresholds = [2, 5, 10]  # Based on RISK_LEVELS

    # Validate input
    if risk_field not in df.columns:
        raise ValueError(f"Risk field '{risk_field}' not found in DataFrame")

    if len(df) == 0:
        logger.warning("Empty DataFrame provided")
        return {
            'total_records': 0,
            'valid_records': 0,
            'null_count': 0,
            'min_risk': None,
            'max_risk': None,
            'mean_risk': None,
            'median_risk': None,
            'std_risk': None,
            'distribution': {},
            'thresholds': sorted(thresholds) if thresholds else [],
            'distribution_percentage': {},
            'risk_level_distribution': {}
        }

    # Get risk values and ensure numeric type
    risk_values = pd.to_numeric(df[risk_field], errors='coerce')

    # Count nulls/invalid values
    null_count = risk_values.isna().sum()
    valid_risk_values = risk_values.dropna()

    if len(valid_risk_values) == 0:
        logger.warning(f"No valid numeric values in risk field '{risk_field}'")
        return {
            'total_records': len(df),
            'valid_records': 0,
            'null_count': null_count,
            'min_risk': None,
            'max_risk': None,
            'mean_risk': None,
            'median_risk': None,
            'std_risk': None,
            'distribution': {},
            'thresholds': sorted(thresholds) if thresholds else [],
            'distribution_percentage': {},
            'risk_level_distribution': {}
        }

    # Basic statistics
    stats = {
        'total_records': len(df),
        'valid_records': len(valid_risk_values),
        'null_count': null_count,
        'min_risk': float(valid_risk_values.min()),
        'max_risk': float(valid_risk_values.max()),
        'mean_risk': float(valid_risk_values.mean()),
        'median_risk': float(valid_risk_values.median()),
        'std_risk': float(valid_risk_values.std()) if len(valid_risk_values) > 1 else 0.0
    }

    # Distribution across thresholds
    if not thresholds:
        stats['distribution'] = {}
        stats['thresholds'] = []
        stats['distribution_percentage'] = {}
        stats['risk_level_distribution'] = {}
        return stats

    sorted_thresholds = sorted(thresholds)

    # Use pd.cut for efficient binning
    # Create bins with explicit edges to handle boundary cases correctly
    bins = [-np.inf] + sorted_thresholds + [np.inf]
    labels = []

    # Create labels for bins
    labels.append(f'<{sorted_thresholds[0]}')
    for i in range(len(sorted_thresholds) - 1):
        labels.append(f'[{sorted_thresholds[i]}, {sorted_thresholds[i + 1]})')
    labels.append(f'>={sorted_thresholds[-1]}')

    # Cut the data into bins
    # right=False means intervals are [a, b) except for the last one
    binned = pd.cut(valid_risk_values, bins=bins, labels=labels, right=False, include_lowest=True)

    # Calculate distribution
    distribution = {}
    distribution_counts = binned.value_counts()

    # Ensure all labels are present in distribution
    for label in labels:
        distribution[label] = int(distribution_counts.get(label, 0))

    stats['distribution'] = distribution
    stats['thresholds'] = sorted_thresholds

    # Calculate percentage of records at each risk level
    percentages = {}
    total_valid = len(valid_risk_values)

    if total_valid > 0:
        for key, count in distribution.items():
            percentages[key] = round(100.0 * count / total_valid, 2)
    else:
        for key in distribution:
            percentages[key] = 0.0

    stats['distribution_percentage'] = percentages

    # Add percentage of null values if any
    if null_count > 0:
        stats['null_percentage'] = round(100.0 * null_count / len(df), 2)

    # Add risk level distribution
    risk_level_counts = {}
    for risk_level, (min_k, max_k) in RISK_LEVELS.items():
        mask = (valid_risk_values >= min_k) & (valid_risk_values < max_k)
        risk_level_counts[risk_level] = int(mask.sum())  # type: ignore

    stats['risk_level_distribution'] = risk_level_counts

    return stats


def get_privacy_recommendations(risk_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate privacy recommendations based on risk statistics.

    Parameters:
    -----------
    risk_stats : Dict[str, Any]
        Risk statistics from get_risk_statistics()

    Returns:
    --------
    Dict[str, Any]
        Recommendations including:
        - suggested_privacy_level: Recommended privacy level
        - suggested_strategies: List of recommended strategies
        - reasoning: Explanation for recommendations
    """
    recommendations = {
        "suggested_privacy_level": "MEDIUM",
        "suggested_strategies": [],
        "reasoning": []
    }

    # Analyze risk distribution
    if risk_stats.get('valid_records', 0) == 0:
        recommendations["reasoning"].append("No valid risk data available")
        return recommendations

    risk_dist = risk_stats.get('risk_level_distribution', {})
    total_valid = risk_stats['valid_records']

    # Calculate percentages
    very_high_pct = risk_dist.get('VERY_HIGH', 0) / total_valid * 100
    high_pct = risk_dist.get('HIGH', 0) / total_valid * 100

    # Determine privacy level based on risk distribution
    if very_high_pct > 20:
        recommendations["suggested_privacy_level"] = "VERY_HIGH"
        recommendations["reasoning"].append(
            f"{very_high_pct:.1f}% of records have very high risk (k<2)"
        )
        recommendations["suggested_strategies"].extend([
            "Use suppression for high-risk records",
            "Apply strong generalization",
            "Consider removing extremely vulnerable records"
        ])

    elif very_high_pct > 10 or high_pct > 30:
        recommendations["suggested_privacy_level"] = "HIGH"
        recommendations["reasoning"].append(
            f"{very_high_pct + high_pct:.1f}% of records have high or very high risk"
        )
        recommendations["suggested_strategies"].extend([
            "Apply conservative anonymization strategies",
            "Use statistical replacement for vulnerable records",
            "Increase generalization levels"
        ])

    elif high_pct > 20:
        recommendations["suggested_privacy_level"] = "MEDIUM"
        recommendations["reasoning"].append(
            f"{high_pct:.1f}% of records have moderate risk"
        )
        recommendations["suggested_strategies"].extend([
            "Use adaptive anonymization based on risk levels",
            "Apply moderate generalization",
            "Preserve utility where possible"
        ])

    else:
        recommendations["suggested_privacy_level"] = "LOW"
        recommendations["reasoning"].append(
            "Most records have low privacy risk"
        )
        recommendations["suggested_strategies"].extend([
            "Use minimal anonymization to preserve utility",
            "Focus on specific high-risk records only",
            "Consider noise addition instead of suppression"
        ])

    # Add specific recommendations based on mean risk
    mean_risk = risk_stats.get('mean_risk', 0)
    if mean_risk < 5:
        recommendations["suggested_strategies"].append(
            f"Mean k-anonymity ({mean_risk:.1f}) is below threshold"
        )

    return recommendations