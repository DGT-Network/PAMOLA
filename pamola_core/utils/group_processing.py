"""
PAMOLA.CORE - Group Processing Utilities for Anonymization
---------------------------------------------------------
This module provides utility functions for processing and analyzing groups
in datasets for anonymization techniques such as k-anonymity, l-diversity,
and t-closeness.

Key features:
- Calculation of group sizes based on quasi-identifiers
- Adaptive k-threshold application for different groups
- Input validation for anonymization operations
- Support for large datasets through Dask integration
- Performance optimizations for group operations

These utilities are used by various anonymization processors to ensure
efficient and consistent group processing across different privacy models.

(C) 2024 Realm Inveo Inc. and DGT Network Inc.

This software is licensed under the BSD 3-Clause License.
For details, see the LICENSE file or visit:

    https://opensource.org/licenses/BSD-3-Clause
    https://github.com/DGT-Network/PAMOLA/blob/main/LICENSE

Author: Realm Inveo Inc. & DGT Network Inc.
"""

import logging
from typing import Dict, List, Tuple

import pandas as pd
from dask import dataframe as dd

# Configure logging
logger = logging.getLogger(__name__)


def compute_group_sizes(data: pd.DataFrame, quasi_identifiers: List[str], use_dask: bool = False) -> pd.Series:
    """
    Computes the size of each unique group based on quasi-identifiers.

    This function is crucial for anonymization techniques like k-anonymity,
    as it determines how many records share the same combination of
    quasi-identifier values.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing quasi-identifiers.
    quasi_identifiers : list[str]
        List of column names defining groups.
    use_dask : bool, optional
        If True, enables parallel processing with Dask (default: False).

    Returns:
    --------
    pd.Series
        A series mapping each group to its size.
    """
    # Validate inputs
    validate_anonymity_inputs(data, quasi_identifiers, 1)  # k=1 is a placeholder here

    if use_dask:
        logger.info("Using Dask for parallel computation of group sizes...")
        if not isinstance(data, dd.DataFrame):
            data = dd.from_pandas(data, npartitions=4)
        return data.groupby(quasi_identifiers).size().compute()

    return data.groupby(quasi_identifiers).size()


def adaptive_k_lookup(group_sizes: pd.Series, default_k: int, adaptive_k: Dict = None) -> pd.Series:
    """
    Adjusts k-threshold dynamically for different quasi-identifier groups.

    This allows for varying levels of anonymity based on the sensitivity
    or characteristics of different groups in the dataset.

    Parameters:
    -----------
    group_sizes : pd.Series
        Series mapping each group to its count.
    default_k : int
        The default k value if no adaptive rules apply.
    adaptive_k : dict, optional
        Dictionary defining custom k-values for specific groups.

    Returns:
    --------
    pd.Series
        A series mapping each group to its adaptive k-value.
    """
    if adaptive_k is None:
        return pd.Series(default_k, index=group_sizes.index)

    return group_sizes.index.to_frame().apply(lambda row: adaptive_k.get(tuple(row), default_k), axis=1)


def validate_anonymity_inputs(data: pd.DataFrame, quasi_identifiers: List[str], k: int) -> None:
    """
    Validates input data and parameters for anonymity operations.

    This function performs comprehensive checks to ensure that the inputs
    to anonymization functions are valid, helping to prevent errors
    and provide clear error messages.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to validate.
    quasi_identifiers : list[str]
        List of quasi-identifiers to validate.
    k : int
        The k value to validate.

    Raises:
    -------
    TypeError
        If data is not a pandas DataFrame or dask DataFrame.
    ValueError
        If quasi-identifiers list is empty or contains invalid column names,
        or if k is less than 1.
    """
    # Check if inputs are None
    if data is None:
        raise ValueError("Input data cannot be None")
    if quasi_identifiers is None:
        raise ValueError("Quasi-identifiers list cannot be None")

    # Check data type
    if not isinstance(data, pd.DataFrame) and not isinstance(data, dd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame or dask DataFrame")

    # Check quasi-identifiers
    if not quasi_identifiers:
        raise ValueError("At least one quasi-identifier must be provided")

    # Check if all quasi-identifiers exist in data
    missing_columns = [col for col in quasi_identifiers if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Quasi-identifiers not found in data: {missing_columns}")

    # Check if k is valid
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")


def optimize_memory_usage(data: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
    """
    Optimizes memory usage by converting appropriate columns to more efficient types.

    This is particularly useful for large datasets where memory optimization
    can significantly improve performance.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to optimize.
    quasi_identifiers : list[str]
        List of quasi-identifiers to focus on for optimization.

    Returns:
    --------
    pd.DataFrame
        The memory-optimized dataset.
    """
    # Make a copy to avoid modifying the original
    result = data.copy()

    for col in quasi_identifiers:
        if col in result.columns:
            # Convert string columns to categorical
            if result[col].dtype == 'object':
                result[col] = result[col].astype('category')

            # Downcast integer columns where possible
            elif pd.api.types.is_integer_dtype(result[col].dtype):
                result[col] = pd.to_numeric(result[col], downcast='integer')

            # Downcast float columns where possible
            elif pd.api.types.is_float_dtype(result[col].dtype):
                result[col] = pd.to_numeric(result[col], downcast='float')

    logger.info(
        f"Memory usage optimized: {data.memory_usage().sum() / 1e6:.2f} MB → {result.memory_usage().sum() / 1e6:.2f} MB")
    return result


def identify_unique_groups(data: pd.DataFrame, quasi_identifiers: List[str], max_groups: int = 100) -> Dict[Tuple, int]:
    """
    Identifies and returns the most common unique groups based on quasi-identifiers.

    This is useful for analyzing the distribution of quasi-identifier combinations
    and understanding potential privacy risks.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset to analyze.
    quasi_identifiers : list[str]
        List of column names defining groups.
    max_groups : int, optional
        Maximum number of groups to return (default: 100).

    Returns:
    --------
    dict
        Dictionary mapping group values to counts, sorted by count (descending).
    """
    # Validate inputs
    validate_anonymity_inputs(data, quasi_identifiers, 1)  # k=1 is a placeholder here

    # Group data and get counts
    group_counts = data.groupby(quasi_identifiers).size().sort_values(ascending=False)

    # Convert to dictionary, limiting to max_groups
    result = {tuple(idx): count for idx, count in zip(group_counts.index.to_frame().values, group_counts.values)}

    # Trim to maximum number of groups
    if len(result) > max_groups:
        result = dict(list(result.items())[:max_groups])

    return result