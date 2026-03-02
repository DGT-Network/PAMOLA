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

import hashlib
from typing import Any, Optional, Union, Dict
import dask.dataframe as dd
import numpy as np
import pandas as pd
import inspect
import logging
import json
from datetime import datetime
from pamola_core.utils.ops.op_data_processing import force_garbage_collection
from pamola_core.utils.ops.op_result import (
    OperationArtifact,
    OperationResult,
    OperationStatus,
)
from pamola_core.errors.exceptions import TypeValidationError

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


def cleanup_memory(instance: Optional[Any] = None, force_gc: bool = True) -> None:
    """
    Cleans up memory by clearing specific attributes of the provided instance.
    This function performs the following actions:
    - Clears the `operation_cache` attribute if it exists.
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

        # Clear error handler
        if hasattr(instance, "error_handler"):
            instance.error_handler = None

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


def stable_json(obj) -> str:
    """
    Convert object to stable sorted JSON.
    Deterministic serialization for hashing.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def blake(data: bytes) -> str:
    """
    Fast, modern hashing using BLAKE2b.
    """
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def safe_numpy_bytes(arr: np.ndarray) -> bytes:
    """
    Convert numpy array to deterministic bytes for hashing.
    """
    if pd.api.types.is_string_dtype(arr):
        return pd.util.hash_pandas_object(pd.Series(arr), index=False).values.tobytes()
    return arr.tobytes()


def smart_sample_indices(n: int, sample_size: int) -> np.ndarray:
    """
    Generate evenly-spaced sample indices using linspace.
    Ensures representative sampling across entire dataset.
    """
    if n <= sample_size:
        return np.arange(n)
    return np.linspace(0, n - 1, sample_size, dtype=int)


def get_series_signature(s: pd.Series, sample_size: int = 5000) -> str:
    """
    Generate deterministic signature for pandas Series.
    """
    metadata = {
        "type": "series",
        "length": len(s),
        "dtype": str(s.dtype),
    }

    if s.empty:
        return stable_json(metadata)

    # Smart sampling - evenly distributed across entire series
    indices = smart_sample_indices(len(s), sample_size)
    sample_vals = s.iloc[indices].to_numpy()
    sample_hash = blake(safe_numpy_bytes(sample_vals))

    # Global checksums
    checksums = {"nulls": int(s.isna().sum())}

    # Numeric stats - single pass optimization
    if pd.api.types.is_numeric_dtype(s):
        try:
            stats = s.agg(["sum", "min", "max"])
            _sum = stats["sum"]

            checksums["sum"] = (
                str(_sum) if pd.isna(_sum) or np.isinf(_sum) else round(float(_sum), 8)
            )
            checksums["min"] = str(stats["min"])
            checksums["max"] = str(stats["max"])
        except Exception:
            # Fallback if agg fails
            pass

    return stable_json(
        {"metadata": metadata, "sample_hash": sample_hash, "checksums": checksums}
    )


def get_df_signature(df: pd.DataFrame, sample_size: int = 5000) -> str:
    """
    Generate deterministic signature for pandas DataFrame.

    """
    metadata = {
        "type": "pandas_df",
        "shape": df.shape,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "columns": df.columns.tolist(),
    }

    if df.empty:
        return stable_json(metadata)

    indices = smart_sample_indices(len(df), sample_size)
    sample_df = df.iloc[indices]

    # Concatenate bytes from all columns for comprehensive hash
    sample_bytes = []
    for col in sample_df.columns:
        arr = sample_df[col].to_numpy()
        sample_bytes.append(safe_numpy_bytes(arr))

    sample_hash = blake(b"".join(sample_bytes))

    checksums = {
        "rows": df.shape[0],
    }

    # Null counts - O(rows * cols), but vectorized and fast
    checksums["nulls"] = {col: int(df[col].isna().sum()) for col in df.columns}

    # Numeric stats - OPTIMIZED: single pass for all stats
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        try:
            stats = df[numeric_cols].agg(["sum", "min", "max"]).to_dict()

            checksums["numeric"] = {}
            for col in numeric_cols:
                _sum = stats[col]["sum"]
                checksums["numeric"][col] = {
                    "sum": (
                        str(_sum)
                        if pd.isna(_sum) or np.isinf(_sum)
                        else round(float(_sum), 8)
                    ),
                    "min": str(stats[col]["min"]),
                    "max": str(stats[col]["max"]),
                }
        except Exception:
            # Fallback: skip numeric checksums if agg fails
            pass

    return stable_json(
        {"metadata": metadata, "sample_hash": sample_hash, "checksums": checksums}
    )


def get_dask_df_signature(df: dd.DataFrame, sample_size: int = 5000) -> str:
    """
    Generate deterministic signature for Dask DataFrame.

    Note: Dask doesn't support iloc with array indices efficiently.
    Falls back to head() for sampling to avoid expensive compute.
    This is acceptable since we have global checksums to catch changes.
    """
    metadata = {
        "type": "dask_df",
        "npartitions": df.npartitions,
        "columns": df.columns.tolist(),
        "dtypes": {c: str(df.dtypes[c]) for c in df.columns},
    }

    try:
        sample_df = df.head(sample_size, npartitions=2)
    except Exception:
        return stable_json({"metadata": metadata, "sample_hash": None})

    if len(sample_df) == 0:
        return stable_json({"metadata": metadata})

    # Hash sample (same as pandas)
    sample_bytes = []
    for col in sample_df.columns:
        sample_bytes.append(safe_numpy_bytes(sample_df[col].to_numpy()))
    sample_hash = blake(b"".join(sample_bytes))

    return stable_json(
        {
            "metadata": metadata,
            "sample_hash": sample_hash,
            "checksums": {"npartitions": df.npartitions},
        }
    )


def generate_data_hash(
    data: Union[pd.DataFrame, pd.Series, dd.DataFrame], sample_size: int = 5000
) -> str:
    """
    Generate a deterministic hash for pandas/dask data structures.

    The hash represents:
    - Data structure (shape, dtypes, columns)
    - Sample of actual values (evenly distributed across dataset)
    - Global statistics (null counts, numeric sum/min/max)

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, dd.DataFrame]
        Input data to hash
    sample_size : int, default=5000
        Number of rows to include in sample hash.
        Rows are selected using linspace for even distribution.
    Returns
    -------
    str
        32-character hex hash string (BLAKE2b)
    """
    try:
        if isinstance(data, pd.Series):
            sig = get_series_signature(data, sample_size)
        elif isinstance(data, pd.DataFrame):
            sig = get_df_signature(data, sample_size)
        elif isinstance(data, dd.DataFrame):
            sig = get_dask_df_signature(data, sample_size)
        else:
            raise TypeValidationError(
                f"Unsupported data type: {type(data).__name__}. "
                "Expected pandas.Series, pandas.DataFrame, or dask.dataframe.DataFrame"
            )

        return blake(sig.encode())

    except Exception as e:
        fallback = f"{type(data).__name__}_{getattr(data, '__len__', lambda: 0)()}"
        return blake(fallback.encode())


def build_base_cache(
    parameters: Dict[str, str], result: OperationResult
) -> Dict[str, Any]:
    """
    Build the base cache structure with timestamp, parameters, metrics, and artifacts.
    All artifacts must implement `.to_dict()`. NumPy numeric types in metrics are
    converted to Python floats for JSON compatibility.
    """

    serialized_metrics = {
        k: float(v) if isinstance(v, (np.integer, np.floating)) else v
        for k, v in result.metrics.items()
    }

    artifacts_for_cache = [artifact.to_dict() for artifact in result.artifacts]

    return {
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "status": (
            result.status.name
            if isinstance(result.status, OperationStatus)
            else str(result.status)
        ),
        "metrics": serialized_metrics,
        "error_message": result.error_message,
        "execution_time": result.execution_time,
        "error_trace": result.error_trace,
        "artifacts": artifacts_for_cache,
    }


def get_cache_result(
    result_data: Optional[Dict[str, Any]],
) -> Optional[OperationResult]:
    """
    Retrieve cached result if available and valid.

    Parameters
    ----------
    result_data : Optional[Dict[str, Any]]
        The cached result data dictionary.

    Returns
    -------
    Optional[OperationResult]
        The cached OperationResult if available, otherwise None.
    """
    try:
        if not isinstance(result_data, dict):
            return None

        # Parse status enum safely
        status_str = result_data.get("status", OperationStatus.ERROR.name)
        status = (
            OperationStatus[status_str]
            if isinstance(status_str, str) and status_str in OperationStatus.__members__
            else OperationStatus.ERROR
        )

        # Rebuild artifacts
        artifacts = []
        for art_dict in result_data.get("artifacts", []):
            if isinstance(art_dict, dict):
                try:
                    artifacts.append(
                        OperationArtifact(
                            artifact_type=art_dict.get("type"),
                            path=art_dict.get("path"),
                            description=art_dict.get("description", ""),
                            category=art_dict.get("category", "output"),
                            tags=art_dict.get("tags", []),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to deserialize artifact: {e}")

        # Determine how many artifacts were restored
        artifacts_restored = len(artifacts)

        # Build result
        result = OperationResult(
            status=status,
            artifacts=artifacts,
            metrics=result_data.get("metrics", {}),
            error_message=result_data.get("error_message"),
            execution_time=result_data.get("execution_time"),
            error_trace=result_data.get("error_trace"),
        )
        result.add_metric("cached", True)
        result.add_metric("artifacts_restored", artifacts_restored)

        return result

    except Exception as e:
        logger.warning(f"Failed to load cache: {e}")
        return None
