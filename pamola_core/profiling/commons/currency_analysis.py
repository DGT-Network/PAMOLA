"""
Currency and salary analysis utilities for the anonymization project.

This module provides specialized functions for analyzing currency and salary fields,
including currency conversion, distribution analysis, and specialized statistics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from pamola_core.profiling.commons.currency_utils import analyze_currency_stats
from pamola_core.utils.logging import configure_logging
from pamola_core.profiling.commons.numeric_utils import (
    calculate_extended_stats,
    calculate_percentiles,
    calculate_histogram,
    detect_outliers,
    test_normality,
    create_empty_stats,
)
from pamola_core.profiling.commons.data_types import DataType

# Configure logger using the custom logging utility
logger = configure_logging(level=logging.INFO)


def analyze_currency_field(
        df: pd.DataFrame,
        value_field: str,
        currency_field: Optional[str] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Analyze a currency field, potentially with an accompanying currency code field.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    value_field : str
        The name of the field with currency values
    currency_field : str, optional
        The name of the field with currency codes
    **kwargs : dict
        Additional parameters:
        - exchange_rates: Dict[str, float], optional
          Dictionary of currency exchange rates for conversion
        - base_currency: str, optional
          Base currency for conversion (default: 'RUB')
        - perform_conversion: bool, optional
          Whether to perform currency conversion (default: True if exchange_rates provided)

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    logger.info(f"Analyzing currency field: {value_field}")

    if value_field not in df.columns:
        return {'error': f"Field {value_field} not found in DataFrame"}

    # Import NumericAnalyzer for basic analysis
    from pamola_core.profiling.analyzers.numeric import NumericAnalyzer

    # Create analyzer instance
    analyzer = NumericAnalyzer()

    # Basic analysis of the value field
    basic_result = analyzer.analyze(df, value_field, **kwargs)
    results = basic_result.stats.copy()

    # Add currency distribution if a currency field is provided
    if currency_field and currency_field in df.columns:
        # Use utility function for currency analysis
        exchange_rates = kwargs.get('exchange_rates')
        base_currency = kwargs.get('base_currency', 'RUB')

        currency_results = analyze_currency_stats(
            df,
            value_field,
            currency_field,
            exchange_rates,
            base_currency
        )

        # Merge the results
        results.update(currency_results)

    return results


def analyze_salary_field(
        df: pd.DataFrame,
        salary_field: str,
        currency_field: Optional[str] = None,
        **kwargs
) -> Dict[str, Any]:
    """
    Specialized analysis for salary fields.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the data
    salary_field : str
        The name of the field with salary values
    currency_field : str, optional
        The name of the field with currency codes
    **kwargs : dict
        Additional parameters for the analysis

    Returns:
    --------
    Dict[str, Any]
        Results of the analysis
    """
    return analyze_currency_field(df, salary_field, currency_field, **kwargs)


def convert_currencies(
        df: pd.DataFrame,
        value_field: str,
        currency_field: str,
        exchange_rates: Dict[str, float],
        base_currency: str = 'RUB'
) -> pd.Series:
    """
    Convert currency values to a base currency using exchange rates.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    value_field : str
        Name of the field with currency values
    currency_field : str
        Name of the field with currency codes
    exchange_rates : Dict[str, float]
        Dictionary mapping currency codes to exchange rates
    base_currency : str, optional
        Base currency for conversion (default: 'RUB')

    Returns:
    --------
    pd.Series
        Series with converted values
    """
    # Create series with converted values
    converted_values = []

    for idx, row in df.iterrows():
        value = row[value_field]
        currency = row[currency_field]

        if pd.isna(value) or pd.isna(currency):
            continue

        if currency == base_currency:
            converted_values.append(value)
        elif currency in exchange_rates:
            converted_value = value * exchange_rates[currency]
            converted_values.append(converted_value)

    if converted_values:
        return pd.Series(converted_values)
    else:
        return pd.Series()