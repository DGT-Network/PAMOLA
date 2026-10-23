"""
PAMOLA.CORE - Utility Functions
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
    Parameters
    ----------
        instance (Optional[Any]): The instance from which to clear attributes.
                                   If None, only class-level attributes will be cleared.
        force_gc (bool): A flag indicating whether to force garbage collection.
                         Defaults to True.
    Returns
    -------
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


#: Rows hashed per step. Bounds the temporary memory of the streaming hash
#: (one chunk of one column at a time), whatever the size of the frame.
_HASH_CHUNK_ROWS = 200_000


def _update_with_series(hasher: "hashlib.blake2b", s: pd.Series) -> None:
    """Feed every value of ``s`` (and nothing else) into ``hasher``, chunk by chunk."""
    for start in range(0, len(s), _HASH_CHUNK_ROWS):
        chunk = s.iloc[start : start + _HASH_CHUNK_ROWS]
        try:
            hashed = pd.util.hash_pandas_object(chunk, index=False, categorize=True)
        except TypeError:
            # Unhashable cell values (lists, dicts): hash their text form.
            hashed = pd.util.hash_pandas_object(
                chunk.astype(str), index=False, categorize=True
            )
        hasher.update(hashed.to_numpy().tobytes())


def _update_with_index(hasher: "hashlib.blake2b", index: pd.Index) -> None:
    for start in range(0, len(index), _HASH_CHUNK_ROWS):
        chunk = index[start : start + _HASH_CHUNK_ROWS]
        hashed = pd.util.hash_pandas_object(chunk, index=False)
        hasher.update(hashed.to_numpy().tobytes())


def _update_with_frame(hasher: "hashlib.blake2b", df: pd.DataFrame) -> None:
    hasher.update(
        stable_json(
            {
                "type": "pandas_df",
                "shape": list(df.shape),
                "columns": [str(c) for c in df.columns],
                "dtypes": [str(t) for t in df.dtypes],
                "index_dtype": str(df.index.dtype),
            }
        ).encode()
    )
    _update_with_index(hasher, df.index)
    for position in range(df.shape[1]):
        _update_with_series(hasher, df.iloc[:, position])


def generate_data_hash(
    data: Union[pd.DataFrame, pd.Series, dd.DataFrame], sample_size: int = 5000
) -> str:
    """
    Generate a deterministic hash that identifies the exact contents of the data.

    The hash covers the structure (shape, column names, dtypes), the index labels
    and **every value**. It is computed as a stream over fixed-size chunks of one
    column at a time, so the temporary memory is bounded and no full copy of the
    frame is made. Changing any single cell, or reordering rows or index labels,
    changes the hash. It is used as cache identity, where a weaker fingerprint
    returns a stale result for different data.

    Parameters
    ----------
    data : Union[pd.DataFrame, pd.Series, dd.DataFrame]
        Input data to hash. A Dask frame is hashed partition by partition.
    sample_size : int, default=5000
        Ignored; kept so existing callers keep working. The earlier sampled
        fingerprint is gone because it missed changes outside the sample.

    Returns
    -------
    str
        64-character hex digest (BLAKE2b-256).

    Raises
    ------
    TypeValidationError
        For an unsupported type.
    Exception
        Whatever hashing the values raised. There is deliberately no fallback to
        a type-and-length fingerprint: callers must disable caching for that call.
    """
    hasher = hashlib.blake2b(digest_size=32)
    if isinstance(data, pd.Series):
        hasher.update(
            stable_json(
                {"type": "series", "length": len(data), "dtype": str(data.dtype),
                 "name": str(data.name)}
            ).encode()
        )
        _update_with_index(hasher, data.index)
        _update_with_series(hasher, data)
    elif isinstance(data, pd.DataFrame):
        _update_with_frame(hasher, data)
    elif isinstance(data, dd.DataFrame):
        hasher.update(
            stable_json(
                {"type": "dask_df", "npartitions": data.npartitions,
                 "columns": [str(c) for c in data.columns]}
            ).encode()
        )
        for i in range(data.npartitions):
            _update_with_frame(hasher, data.get_partition(i).compute())
    else:
        raise TypeValidationError(
            param_name="data",
            expected_type="pandas.Series, pandas.DataFrame or dask.dataframe.DataFrame",
            actual_type=type(data).__name__,
            message=(
                f"Unsupported data type: {type(data).__name__}. "
                "Expected pandas.Series, pandas.DataFrame, or dask.dataframe.DataFrame"
            ),
        )
    return hasher.hexdigest()


#: Parameter names whose values are secrets and must never be persisted or logged.
_SENSITIVE_NAMES = frozenset(
    {
        "encryption_key", "password", "token", "secret", "api_key", "salt",
        "pepper", "passphrase", "private_key", "key",
    }
)
_SENSITIVE_SUFFIXES = (
    "_key", "_password", "_secret", "_token", "_passphrase", "_salt", "_pepper",
)
_REDACTED = "*REDACTED*"


def is_sensitive_parameter(name: Any) -> bool:
    """True if a parameter called ``name`` carries a secret."""
    lowered = str(name).lower()
    return lowered in _SENSITIVE_NAMES or lowered.endswith(_SENSITIVE_SUFFIXES)


def redact_sensitive_parameters(value: Any, extra_keys: Any = ()) -> Any:
    """Copy of ``value`` with the values of secret-named keys replaced.

    Applied before any parameters are serialized (cache metadata, logs). Works
    recursively through dicts, lists and tuples; ``extra_keys`` adds names an
    operation's config declares as sensitive (``OperationConfig.SENSITIVE_KEYS``).
    """
    extra = set(extra_keys)
    if isinstance(value, dict):
        return {
            k: (
                _REDACTED
                if (k in extra or is_sensitive_parameter(k)) and v is not None
                else redact_sensitive_parameters(v, extra)
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_parameters(v, extra) for v in value]
    return value


def secret_identity(secret: Any) -> Optional[str]:
    """Non-secret stand-in for a secret that must still influence a cache key.

    A truncated BLAKE2b digest: stable for the same secret, different for another,
    and not the secret. ``None`` stays ``None``.
    """
    if secret is None:
        return None
    return hashlib.blake2b(
        str(secret).encode("utf-8"), digest_size=16, person=b"pamola-cache"
    ).hexdigest()


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
        "parameters": redact_sensitive_parameters(parameters),
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
