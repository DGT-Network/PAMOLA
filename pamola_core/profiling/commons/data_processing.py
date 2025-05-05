"""
Data processing utilities for the HHR anonymization project.

This module provides functions for preparing and preprocessing data before analysis,
including handling of missing values, type conversion, and chunked processing.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable

import pandas as pd
import numpy as np
from pamola_core.utils.logging import configure_logging
from pamola_core.utils.progress import ProgressTracker, process_dataframe_in_chunks

# Configure logger using the custom logging utility
logger = configure_logging(level=logging.INFO)


def prepare_numeric_data(
        df: pd.DataFrame,
        field_name: str
) -> Tuple[pd.Series, int, int]:
    """
    Prepare numeric data for analysis, handling conversions and missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    field_name : str
        Name of the field to prepare

    Returns:
    --------
    Tuple[pd.Series, int, int]
        Tuple containing:
        - Series of prepared numeric data
        - Count of null values
        - Count of non-null values
    """
    # Check if field exists
    if field_name not in df.columns:
        return pd.Series(), 0, 0

    # Get total rows
    total_rows = len(df)

    # Count null values
    null_count = df[field_name].isna().sum()
    non_null_count = total_rows - null_count

    # Convert to numeric, coercing errors to NaN
    valid_data = pd.to_numeric(df[field_name].dropna(), errors='coerce')

    # Filter out NaN values that may have been introduced by to_numeric
    valid_data = valid_data.dropna()

    return valid_data, null_count, non_null_count


def prepare_field_for_analysis(
        df: pd.DataFrame,
        field_name: str
) -> Tuple[pd.Series, str]:
    """
    Prepare a field for analysis and infer its data type.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the data
    field_name : str
        Name of the field to prepare

    Returns:
    --------
    Tuple[pd.Series, str]
        Tuple containing:
        - Series of prepared data
        - Inferred data type
    """
    from core.profiling.commons.data_types import DataType

    # Check if field exists
    if field_name not in df.columns:
        return pd.Series(), DataType.UNKNOWN.value

    # Extract series
    series = df[field_name]

    # Infer data type
    if pd.api.types.is_numeric_dtype(series):
        return series, DataType.NUMERIC.value
    elif pd.api.types.is_datetime64_any_dtype(series):
        return series, DataType.DATE.value
    elif series.dtype == 'object':
        # Try to infer more specific type for object series
        if series.str.contains('@', na=False).any():
            return series, DataType.EMAIL.value
        # Add more type inference logic as needed
        return series, DataType.TEXT.value
    else:
        return series, DataType.UNKNOWN.value


def handle_large_dataframe(
        df: pd.DataFrame,
        field_name: str,
        operation: Callable,
        chunk_size: int = 10000,
        **kwargs
) -> Dict[str, Any]:
    """
    Handle large dataframes by processing in chunks.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    field_name : str
        Name of the field to analyze
    operation : Callable
        Function to apply to each chunk
    chunk_size : int
        Size of chunks for processing
    **kwargs : dict
        Additional parameters for the operation

    Returns:
    --------
    Dict[str, Any]
        Combined results of the chunked processing
    """
    from core.profiling.commons.numeric_utils import combine_chunk_results

    # Create DataFrame with only the needed field
    field_df = df[[field_name]].copy()

    # Process data in chunks using the utility from progress.py
    description = f"Processing {field_name} in chunks"

    # Get the function from process_dataframe_in_chunks
    from core.utils.progress import process_dataframe_in_chunks

    chunk_results = process_dataframe_in_chunks(
        field_df,
        lambda chunk_df: operation(chunk_df, field_name, **kwargs),
        description,
        chunk_size
    )

    # Combine chunk results
    return combine_chunk_results(chunk_results)