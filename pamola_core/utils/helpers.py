"""
PAMOLA.CORE - Utility Functions
----------------------------------------------------
Module: Helpers
Description:
    Collection of utility functions for common operations such as:
    - Filtering keyword arguments based on function signatures
    - Extracting characteristics from pandas and Dask data structures
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

from typing import Dict, List, Any, Optional, Tuple
import dask.dataframe as dd
import pandas as pd
import inspect


def filter_used_kwargs(kwargs: dict, func) -> dict:
    """
    Remove keys from kwargs that conflict with the named parameters of the given function.

    :param kwargs: A dictionary of keyword arguments to filter.
    :param func: The target function or method to check against.
    :return: A filtered kwargs dictionary excluding keys that match the function's parameters.
    """
    used_keys = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k not in used_keys}


def get_df_signature(df: pd.DataFrame, sample_size: int = 5000) -> str:
    """
    Get signature for DataFrame optimized for PETs caching.
    Combines sample hash with global checksums for accuracy.
    """
    metadata = {
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in sorted(df.dtypes.items())},
        "columns": sorted(df.columns.tolist()),
    }

    if df.empty:
        return str(metadata)

    # Sample hash
    sample_df = df.head(sample_size)
    sample_hash = pd.util.hash_pandas_object(sample_df, index=False)
    sample_hash_str = sample_hash.values.tobytes().hex()

    # Global checksums
    checksums = []

    # 1. Total row count
    checksums.append(f"rows:{df.shape[0]}")

    # 2. Null count per column
    null_counts = df.isna().sum().to_dict()
    null_sig = "_".join(
        [f"{col}:{count}" for col, count in sorted(null_counts.items())]
    )
    checksums.append(f"nulls:{null_sig}")

    # 3. Numeric columns: sum
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        # Round to avoid float precision issues
        numeric_sums = df[numeric_cols].sum().round(6).to_dict()
        sum_sig = "_".join(
            [f"{col}:{val}" for col, val in sorted(numeric_sums.items())]
        )
        checksums.append(f"sums:{sum_sig}")

    checksum_str = "|".join(checksums)

    return f"{metadata}_{sample_hash_str}_{checksum_str}"


def get_dask_df_signature(df: dd.DataFrame, sample_size: int = 5000) -> str:
    """
    Get signature for Dask DataFrame optimized for PETs caching.
    """
    metadata = {
        "type": "dask",
        "npartitions": df.npartitions,
        "dtypes": {col: str(dtype) for col, dtype in sorted(df.dtypes.items())},
        "columns": sorted(df.columns.tolist()),
    }

    sample_df = df.head(sample_size, npartitions=-1)

    if sample_df.empty:
        return str(metadata)

    # Sample hash
    sample_hash = pd.util.hash_pandas_object(sample_df, index=False)
    sample_hash_str = sample_hash.values.tobytes().hex()

    # For Dask: global checksums would trigger compute() which is expensive
    # So we rely more on sample + metadata
    # But we can add partition info
    checksums = [f"nparts:{df.npartitions}"]
    checksum_str = "|".join(checksums)

    return f"{metadata}_{sample_hash_str}_{checksum_str}"


def get_series_signature(s: pd.Series, sample_size: int = 5000) -> str:
    """
    Get signature for pandas Series optimized for PETs caching.
    """
    metadata = {
        "length": len(s),
        "dtype": str(s.dtype),
    }

    if s.empty:
        return str(metadata)

    # Sample hash
    sample_s = s.head(sample_size)
    sample_hash = pd.util.hash_pandas_object(sample_s, index=False)
    sample_hash_str = sample_hash.values.tobytes().hex()

    # Global checksums
    checksums = [
        f"nulls:{s.isna().sum()}",
    ]

    if pd.api.types.is_numeric_dtype(s):
        checksums.append(f"sum:{round(s.sum(), 6)}")

    checksum_str = "|".join(checksums)

    return f"{metadata}_{sample_hash_str}_{checksum_str}"
