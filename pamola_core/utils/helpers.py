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
import logging
from pamola_core.utils.ops.op_data_processing import force_garbage_collection

# Configure logger
logger = logging.getLogger(__name__)


def filter_used_kwargs(kwargs: dict, func) -> dict:
    """
    Remove keys from kwargs that conflict with the named parameters of the given function.

    :param kwargs: A dictionary of keyword arguments to filter.
    :param func: The target function or method to check against.
    :return: A filtered kwargs dictionary excluding keys that match the function's parameters.
    """
    used_keys = set(inspect.signature(func).parameters)
    return {k: v for k, v in kwargs.items() if k not in used_keys}

def cleanup_memory(
        instance: Optional[Any] = None,
        force_gc: bool = True
    ) -> None:
    """
        Cleans up memory by clearing specific attributes of the provided instance.
        This function performs the following actions:
        - Clears the `operation_cache` attribute if it exists.
        - Resets the `process_kwargs` attribute to an empty dictionary if it exists.
        - Clears the `filter_mask` attribute if it exists.
        - Additionally, it removes any attributes that start with `_temp_` from the instance.
        - If `force_gc` is set to True, it triggers garbage collection.
        Args:
            instance (Optional[Any]): The instance from which to clear attributes. 
                                       If None, only class-level attributes will be cleared.
            force_gc (bool): A flag indicating whether to force garbage collection.
                             Defaults to True.
        Returns:
            None: This function does not return any value.
    """
    try:
        if instance is None:
            return
        
        # Clear operation cache
        if hasattr(instance, "operation_cache"):
            instance.operation_cache = None

        # Clear process kwargs
        if hasattr(instance, "process_kwargs"):
            instance.process_kwargs = {}

        # Clear filter mask
        if hasattr(instance, "filter_mask"):
            instance.filter_mask = None

        # Clear original dataframe cache
        if hasattr(instance, "_original_df"):
            instance._original_df = None

        # Additional cleanup for any temporary attributes
        for attr_name in list(vars(instance).keys()):
            if attr_name.startswith("_temp_"):
                delattr(instance, attr_name)
        
        # Force GC if explicitly requested
        if force_gc:
            force_garbage_collection()
    except Exception as e:
        logger.warning(f"Error cleanup_memory: {e}")


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
