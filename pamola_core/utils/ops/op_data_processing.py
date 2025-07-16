"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Data Processing Utilities
Package:       pamola_core.utils.ops
Version:       2.0.0
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause

Description:
  This module provides essential data processing utilities for all operation types
  within the PAMOLA.CORE framework. It focuses on basic DataFrame transformations,
  memory optimization, and chunking capabilities that are commonly needed across
  different operations.

Key Features:
  - Memory-efficient DataFrame type optimization
  - Simple chunk generation for large datasets
  - Basic null value handling
  - Type conversion utilities

Design Principles:
  - Keep it simple and focused
  - No complex analytics or metrics
  - No performance benchmarking
  - Let operations handle their own validation
  - Minimize external dependencies
"""

import gc
import logging
from typing import Any, Dict, Generator, Optional, Tuple, Union
from typing import Literal

import numpy as np
import pandas as pd


# Configure module logger
logger = logging.getLogger(__name__)


def optimize_dataframe_dtypes(
    df: pd.DataFrame,
    categorical_threshold: float = 0.5,
    downcast_integers: bool = True,
    downcast_floats: bool = True,
    inplace: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Optimize DataFrame data types to reduce memory usage.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize
    categorical_threshold : float, optional
        Ratio threshold for converting string columns to categorical (default: 0.5)
    downcast_integers : bool, optional
        Whether to downcast integer types (default: True)
    downcast_floats : bool, optional
        Whether to downcast float types (default: True)
    inplace : bool, optional
        Whether to modify the DataFrame in place (default: False)

    Returns:
    --------
    Tuple[pd.DataFrame, Dict[str, Any]]
        (optimized_dataframe, optimization_info)
    """
    # Work on a copy if not in-place
    result_df = df if inplace else df.copy()

    # Calculate initial memory usage
    memory_before = result_df.memory_usage(deep=True).sum() / 1024**2  # MB

    # Track changes
    type_changes = {}

    for column in result_df.columns:
        original_dtype = result_df[column].dtype
        column_data = result_df[column]

        try:
            # Handle integers
            if pd.api.types.is_integer_dtype(column_data) and downcast_integers:
                optimized = pd.to_numeric(column_data, downcast="integer")
                if optimized.dtype != original_dtype:
                    result_df[column] = optimized
                    type_changes[column] = f"{original_dtype} -> {optimized.dtype}"

            # Handle floats
            elif pd.api.types.is_float_dtype(column_data) and downcast_floats:
                optimized = pd.to_numeric(column_data, downcast="float")
                if optimized.dtype != original_dtype:
                    result_df[column] = optimized
                    type_changes[column] = f"{original_dtype} -> {optimized.dtype}"

            # Handle objects (potential strings)
            elif pd.api.types.is_object_dtype(column_data):
                non_null_count = column_data.count()
                if non_null_count > 0:
                    unique_ratio = column_data.nunique() / non_null_count
                    if unique_ratio < categorical_threshold:
                        result_df[column] = column_data.astype("category")
                        type_changes[column] = f"{original_dtype} -> category"

        except Exception as e:
            logger.debug(f"Could not optimize column {column}: {str(e)}")
            continue

    # Calculate final memory usage
    memory_after = result_df.memory_usage(deep=True).sum() / 1024**2  # MB

    optimization_info = {
        "memory_before_mb": round(memory_before, 2),
        "memory_after_mb": round(memory_after, 2),
        "memory_saved_mb": round(memory_before - memory_after, 2),
        "memory_saved_percent": (
            round((1 - memory_after / memory_before) * 100, 2)
            if memory_before > 0
            else 0
        ),
        "columns_optimized": len(type_changes),
        "type_changes": type_changes,
    }

    return result_df, optimization_info


def get_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000
) -> Generator[pd.DataFrame, None, None]:
    """
    Generate DataFrame chunks for memory-efficient processing.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to chunk
    chunk_size : int, optional
        Size of each chunk (default: 10000)

    Yields:
    -------
    pd.DataFrame
        Chunks of the original DataFrame
    """
    if len(df) == 0:
        return

    for i in range(0, len(df), chunk_size):
        yield df.iloc[i : i + chunk_size].copy()


def process_null_values(
    series: pd.Series, strategy: str = "preserve", fill_value: Any = None
) -> pd.Series:
    """
    Process null values in a Series according to strategy.

    Parameters:
    -----------
    series : pd.Series
        Series to process
    strategy : str, optional
        Strategy for handling nulls: "preserve", "drop", "fill" (default: "preserve")
    fill_value : Any, optional
        Value to use when strategy is "fill"

    Returns:
    --------
    pd.Series
        Processed series
    """
    if strategy == "preserve":
        return series
    elif strategy == "drop":
        return series.dropna()
    elif strategy == "fill":
        if fill_value is None:
            # Use appropriate default based on dtype
            if pd.api.types.is_numeric_dtype(series):
                fill_value = 0
            else:
                fill_value = ""
        return series.fillna(fill_value)
    else:
        raise ValueError(f"Unknown null strategy: {strategy}")


def safe_convert_to_numeric(
    series: pd.Series, errors: Literal["ignore", "raise", "coerce"] = "coerce"
) -> pd.Series:
    """
    Safely convert series to numeric, handling errors gracefully.

    Parameters:
    -----------
    series : pd.Series
        Series to convert
    errors : Literal['coerce', 'ignore', 'raise'], optional
        How to handle parsing errors: 'coerce', 'ignore', 'raise' (default: 'coerce')

    Returns:
    --------
    pd.Series
        Numeric series
    """
    return pd.to_numeric(series, errors=errors)


def get_memory_usage(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get memory usage statistics for a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze

    Returns:
    --------
    Dict[str, float]
        Memory usage information in MB
    """
    total_memory = df.memory_usage(deep=True).sum()

    return {
        "total_mb": round(total_memory / 1024**2, 2),
        "per_row_bytes": round(total_memory / len(df), 2) if len(df) > 0 else 0,
        "per_column_avg_mb": (
            round(total_memory / len(df.columns) / 1024**2, 2)
            if len(df.columns) > 0
            else 0
        ),
    }


def apply_to_column(
    df: pd.DataFrame,
    column: str,
    func: callable,
    result_column: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Apply a function to a DataFrame column with optional result column.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    column : str
        Column to apply function to
    func : callable
        Function to apply
    result_column : Optional[str], optional
        Name for result column (default: None, modifies in place)
    **kwargs
        Additional arguments passed to func

    Returns:
    --------
    pd.DataFrame
        DataFrame with applied transformation
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    result = func(df[column], **kwargs)

    if result_column:
        df[result_column] = result
    else:
        df[column] = result

    return df


def create_sample(
    df: pd.DataFrame,
    n: Optional[int] = None,
    frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Create a sample from DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to sample from
    n : Optional[int], optional
        Number of items to sample
    frac : Optional[float], optional
        Fraction of items to sample
    random_state : int, optional
        Random seed (default: 42)

    Returns:
    --------
    pd.DataFrame
        Sampled DataFrame
    """
    if n is None and frac is None:
        n = min(10000, len(df))  # Default sample size

    return df.sample(n=n, frac=frac, random_state=random_state)


def force_garbage_collection() -> None:
    """
    Force garbage collection to free memory.

    Useful between processing large chunks of data.
    """
    gc.collect()


# Module metadata
__version__ = "2.0.0"
__author__ = "PAMOLA Core Team"
__license__ = "BSD 3-Clause"

# Export main functions
__all__ = [
    "optimize_dataframe_dtypes",
    "get_dataframe_chunks",
    "process_null_values",
    "safe_convert_to_numeric",
    "get_memory_usage",
    "apply_to_column",
    "create_sample",
    "force_garbage_collection",
]
