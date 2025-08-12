"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Data Source Helper Functions
Description: Helper functions for DataSource class
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module contains supplementary functions used by the DataSource class.
These functions are factored out to maintain a clean separation of concerns
and focus on value-added functionality not directly available in other modules.

Key features:
- Schema validation and type compatibility
- Memory optimization recommendations
- DataFrame chunking for memory-efficient processing
- DataFrame analysis and diagnostics
"""

import gc
import os
from logging import Logger
from typing import Dict, Any, List, Optional, Tuple, Generator, Callable

import pandas as pd
import psutil

from pamola_core.utils import logging as custom_logging
from pamola_core.utils.io import optimize_dataframe_memory
from pamola_core.utils.progress import track_operation_safely


def get_system_memory() -> Dict[str, float]:
    """
    Get system memory information.

    Returns:
    --------
    Dict[str, float]
        Dictionary with memory information (total, available, used in GB)
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024 ** 3),
            "available_gb": memory.available / (1024 ** 3),
            "used_gb": memory.used / (1024 ** 3),
            "percent": memory.percent
        }
    except Exception as e:
        logger = custom_logging.get_logger("memory_helpers")
        logger.warning(f"Could not get system memory info: {e}")
        return {
            "total_gb": 0,
            "available_gb": 0,
            "used_gb": 0,
            "percent": 0
        }


def get_process_memory_usage() -> Dict[str, float]:
    """
    Get current process memory usage.

    Returns:
    --------
    Dict[str, float]
        Dictionary with process memory usage (RSS, VMS in MB)
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        }
    except Exception as e:
        logger = custom_logging.get_logger("memory_helpers")
        logger.warning(f"Could not get process memory usage: {e}")
        return {
            "rss_mb": 0,
            "vms_mb": 0
        }


def validate_schema(
        actual_schema: Dict[str, Any],
        expected_schema: Dict[str, Any],
        logger: Optional[Logger] = None
) -> Tuple[bool, List[str]]:
    """
    Validate a DataFrame schema against an expected schema.

    Parameters:
    -----------
    actual_schema : Dict[str, Any]
        Actual schema to validate
    expected_schema : Dict[str, Any]
        Expected schema to validate against
    logger : Logger, optional
        Logger instance

    Returns:
    --------
    Tuple[bool, List[str]]
        (is_valid, error_messages)
    """
    # Initialize logger if not provided
    if logger is None:
        logger = custom_logging.get_logger("schema_helpers")

    errors = []

    # Check if all expected columns exist
    if 'columns' in expected_schema:
        missing_cols = set(expected_schema['columns']) - set(actual_schema['columns'])
        if missing_cols:
            error_msg = f"Missing columns: {', '.join(missing_cols)}"
            logger.warning(error_msg)
            errors.append(error_msg)

    # Check if column types match
    if 'dtypes' in expected_schema:
        for col, dtype in expected_schema['dtypes'].items():
            if col in actual_schema['dtypes']:
                actual_dtype = actual_schema['dtypes'][col]
                if not is_compatible_dtype(actual_dtype, dtype):
                    error_msg = f"Column '{col}' has type '{actual_dtype}', expected '{dtype}'"
                    logger.warning(error_msg)
                    errors.append(error_msg)

    # Advanced schema validation: validate custom constraints
    if 'constraints' in expected_schema:
        for constraint in expected_schema['constraints']:
            if constraint.get('type') == 'non_null':
                col = constraint.get('column')
                if col and col in actual_schema.get('null_counts', {}):
                    null_count = actual_schema['null_counts'][col]
                    if null_count > 0:
                        error_msg = f"Column '{col}' has {null_count} null values, expected no nulls"
                        logger.warning(error_msg)
                        errors.append(error_msg)
            elif constraint.get('type') == 'unique':
                col = constraint.get('column')
                if col and col in actual_schema.get('unique_counts', {}):
                    unique_count = actual_schema['unique_counts'][col]
                    total_count = actual_schema['num_rows']
                    if unique_count < total_count:
                        error_msg = f"Column '{col}' has {unique_count} unique values out of {total_count}, expected all unique"
                        logger.warning(error_msg)
                        errors.append(error_msg)

    is_valid = len(errors) == 0
    logger.info(f"Schema validation: {'Valid' if is_valid else 'Invalid'}")
    if not is_valid:
        logger.debug(f"Validation errors: {', '.join(errors)}")

    return is_valid, errors


def is_compatible_dtype(actual: str, expected: str) -> bool:
    """
    Check if two data types are compatible with enhanced type handling.

    Parameters:
    -----------
    actual : str
        Actual data type
    expected : str
        Expected data type

    Returns:
    --------
    bool
        True if compatible, False otherwise
    """
    # Convert string representations to standard forms
    actual = actual.lower()
    expected = expected.lower()

    # Check for exact match
    if actual == expected:
        return True

    # Define type compatibility categories
    integer_types = {'int', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'}
    float_types = {'float', 'float16', 'float32', 'float64', 'double'}
    numeric_types = integer_types.union(float_types)

    string_types = {'object', 'string', 'str'}

    datetime_types = {'datetime', 'datetime64', 'timestamp', 'datetime64[ns]'}
    date_types = {'date', 'datetime', 'datetime64', 'timestamp'}
    time_types = {'time', 'timedelta', 'timedelta64'}

    boolean_types = {'bool', 'boolean'}

    category_types = {'category', 'categorical'}

    # Extract base type from potential parametrized type (e.g., datetime64[ns] -> datetime64)
    def extract_base_type(type_str):
        return type_str.split('[')[0]

    actual_base = extract_base_type(actual)
    expected_base = extract_base_type(expected)

    # Check numeric compatibility (float can hold int, but not vice versa)
    if actual_base in float_types and expected_base in numeric_types:
        return True
    if actual_base in integer_types and expected_base in integer_types:
        return True

    # Check string compatibility
    if actual_base in string_types and expected_base in string_types:
        return True

    # Check datetime compatibility
    if actual_base in datetime_types and expected_base in date_types:
        return True

    # Check time compatibility
    if actual_base in time_types and expected_base in time_types:
        return True

    # Check boolean compatibility
    if actual_base in boolean_types and expected_base in boolean_types:
        return True

    # Check category compatibility
    if actual_base in category_types and expected_base in category_types:
        return True

    # Special cases
    # String can hold any type when needed
    if actual_base in string_types:
        return True

    # Category can be compared with its underlying type
    if actual_base in category_types and expected_base in string_types:
        return True
    if expected_base in category_types and actual_base in string_types:
        return True

    # Handle numpy vs pandas types with equivalent semantics
    if ('int' in actual_base and 'int' in expected_base) or \
            ('float' in actual_base and 'float' in expected_base) or \
            ('datetime' in actual_base and 'datetime' in expected_base) or \
            ('timedelta' in actual_base and 'timedelta' in expected_base):
        return True

    # Default: not compatible
    return False


def generate_dataframe_chunks(
        df: pd.DataFrame,
        chunk_size: int,
        columns: Optional[List[str]] = None,
        logger: Optional[Logger] = None,
        show_progress: bool = True
) -> Generator[pd.DataFrame, None, None]:
    """
    Generate chunks from a DataFrame for efficient processing.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to chunk
    chunk_size : int
        Size of each chunk
    columns : List[str], optional
        Specific columns to include
    logger : Logger, optional
        Logger instance
    show_progress : bool
        Whether to show progress during chunking

    Yields:
    -------
    pd.DataFrame
        Chunks of the DataFrame
    """
    # Initialize logger if not provided
    if logger is None:
        logger = custom_logging.get_logger("chunk_helpers")

    # Select specific columns if requested
    if columns is not None:
        valid_cols = [col for col in columns if col in df.columns]
        if not valid_cols:
            logger.error(f"None of the requested columns exist in DataFrame")
            return
        df = df[valid_cols]
        logger.debug(f"Selected {len(valid_cols)} columns for chunking")

    # Determine the number of chunks
    num_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.debug(f"Processing DataFrame in {num_chunks} chunks of size {chunk_size}")

    # Use track_operation_safely from progress.py for robust error handling
    with track_operation_safely(
            description="Processing DataFrame in chunks",
            total=num_chunks,
            unit="chunks",
            track_memory=True,
            on_error=lambda e: logger.error(f"Error during chunk processing: {e}")
    ) as progress_tracker:
        # Yield chunks with progress tracking
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))

            # Create chunk - use copy for safety
            chunk_df = df.iloc[start_idx:end_idx].copy()

            # Update progress if tracking
            if progress_tracker and show_progress:
                memory_info = {"memory_mb": "N/A"}

                # Try to get memory usage
                try:
                    # Get memory info
                    process_memory = get_process_memory_usage()
                    memory_mb = process_memory.get('rss_mb', 0)
                    memory_info = {"memory_mb": f"{memory_mb:.1f}MB"}
                except Exception:
                    # Continue without memory info if not available
                    pass

                progress_tracker.update(1, {
                    "chunk": f"{i + 1}/{num_chunks}",
                    "rows": len(chunk_df),
                    **memory_info
                })

            yield chunk_df


def optimize_memory_usage(
        dataframes: Dict[str, pd.DataFrame],
        threshold_percent: float = 80.0,
        release_func: Optional[Callable[[str], bool]] = None,
        logger: Optional[Logger] = None
) -> Dict[str, Any]:
    """
    Analyze and optimize memory usage for multiple DataFrames.

    Parameters:
    -----------
    dataframes : Dict[str, pd.DataFrame]
        Dictionary of named DataFrames to optimize
    threshold_percent : float
        Memory usage threshold to trigger optimization (default: 80%)
    release_func : Callable[[str], bool], optional
        Function to release a DataFrame by name
    logger : Logger, optional
        Logger instance

    Returns:
    --------
    Dict[str, Any]
        Memory optimization results
    """
    # Initialize logger if not provided
    if logger is None:
        logger = custom_logging.get_logger("memory_helpers")

    # Initialize result dictionary
    result = {
        "status": "ok",
        "optimizations": {},
        "released_dataframes": [],
        "initial_memory": None,
        "final_memory": None
    }

    # Get system memory information
    system_memory = None
    try:
        system_memory = get_system_memory()
        result["system_memory"] = system_memory
    except Exception as e:
        logger.warning(f"Could not get system memory info: {e}")

    # Calculate current memory usage for DataFrames
    total_memory_mb = 0
    df_memory_usage = {}

    for name, df in dataframes.items():
        try:
            memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
            df_memory_usage[name] = memory_mb
            total_memory_mb += memory_mb
        except Exception as e:
            logger.warning(f"Could not calculate memory for DataFrame '{name}': {e}")

    # Store initial memory usage
    result["initial_memory"] = {
        "total_mb": total_memory_mb,
        "dataframes": df_memory_usage.copy()
    }

    # Calculate usage percentage if system memory info is available
    memory_percent = None
    if system_memory:
        memory_percent = (total_memory_mb * 100) / (system_memory["total_gb"] * 1024)
        result["initial_memory"]["usage_percent"] = memory_percent

    # Check if optimization is needed
    if memory_percent is None or memory_percent < threshold_percent:
        logger.debug(
            f"Memory usage ({memory_percent:.1f if memory_percent else 'unknown'}%) below threshold ({threshold_percent:.1f}%), no optimization needed")
        result["final_memory"] = result["initial_memory"]
        return result

    # Optimize memory usage
    logger.info(f"Memory usage ({memory_percent:.1f}%) exceeds threshold ({threshold_percent:.1f}%), optimizing")
    optimization_results = {}

    # First pass: Optimize all DataFrames in place using io.py
    for name, df in list(dataframes.items()):
        try:
            # Use io.py's optimize_dataframe_memory
            optimized_df, optim_info = optimize_dataframe_memory(df, inplace=True)
            dataframes[name] = optimized_df
            optimization_results[name] = optim_info

            # Update memory usage tracking
            memory_mb = optimized_df.memory_usage(deep=True).sum() / (1024 * 1024)
            df_memory_usage[name] = memory_mb
        except Exception as e:
            logger.warning(f"Failed to optimize DataFrame '{name}': {e}")

    # Recalculate total memory after optimization
    total_memory_mb = sum(df_memory_usage.values())

    if system_memory:
        memory_percent = (total_memory_mb * 100) / (system_memory["total_gb"] * 1024)
    else:
        memory_percent = None

    # Second pass: If still above threshold and release function provided, release some DataFrames
    released_dfs = []
    if memory_percent and memory_percent > threshold_percent and release_func:
        # Sort DataFrames by size (largest first)
        sorted_dfs = sorted(df_memory_usage.items(), key=lambda x: x[1], reverse=True)

        # Release DataFrames until below threshold
        for name, size_mb in sorted_dfs:
            if memory_percent <= threshold_percent:
                break

            # Try to release the DataFrame
            if release_func(name):
                released_dfs.append(name)
                total_memory_mb -= size_mb

                if system_memory:
                    memory_percent = (total_memory_mb * 100) / (system_memory["total_gb"] * 1024)

                logger.info(f"Released DataFrame '{name}' ({size_mb:.1f}MB)")

    # Force garbage collection
    gc.collect()

    # Update result
    result.update({
        "status": "optimized",
        "optimizations": optimization_results,
        "released_dataframes": released_dfs,
        "final_memory": {
            "total_mb": total_memory_mb,
            "dataframes": {name: size for name, size in df_memory_usage.items() if name not in released_dfs},
            "usage_percent": memory_percent
        }
    })

    return result


def analyze_dataframe(
        df: pd.DataFrame,
        logger: Optional[Logger] = None
) -> Dict[str, Any]:
    """
    Analyze DataFrame structure and provide insights.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to analyze
    logger : Logger, optional
        Logger instance

    Returns:
    --------
    Dict[str, Any]
        Dictionary with DataFrame analysis results
    """
    # Initialize logger if not provided
    if logger is None:
        logger = custom_logging.get_logger("dataframe_helpers")

    # Initialize result
    result = {
        "shape": {
            "rows": len(df),
            "columns": len(df.columns)
        },
        "memory_usage": {
            "total_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "by_column": {}
        },
        "column_types": {},
        "null_counts": {},
        "unique_counts": {},
        "potential_optimizations": [],
        "sample_values": {}
    }

    try:
        # Analyze memory usage by column
        mem_usage = df.memory_usage(deep=True)
        for i, col in enumerate(df.columns):
            col_mem = mem_usage[i + 1] / (1024 * 1024)  # MB
            result["memory_usage"]["by_column"][col] = col_mem

        # Analyze column types and null counts
        for col in df.columns:
            result["column_types"][col] = str(df[col].dtype)
            result["null_counts"][col] = int(df[col].isna().sum())

            # Count unique values (skip for very large columns)
            if len(df) < 100000:
                try:
                    result["unique_counts"][col] = int(df[col].nunique())
                except Exception:
                    result["unique_counts"][col] = None

        # Get sample values from first row if not empty
        if len(df) > 0:
            try:
                sample_row = df.iloc[0].to_dict()
                result["sample_values"] = {k: str(v) for k, v in sample_row.items()}
            except Exception:
                pass

        # Check for potential optimizations

        # 1. Check if object columns could be categorical
        for col in df.select_dtypes(include=['object']).columns:
            if col in result["unique_counts"] and result["unique_counts"][col]:
                unique_ratio = result["unique_counts"][col] / len(df)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    result["potential_optimizations"].append({
                        "column": col,
                        "current_type": "object",
                        "suggested_type": "category",
                        "unique_ratio": unique_ratio,
                        "estimated_savings_mb": result["memory_usage"]["by_column"][col] * 0.8  # Approximate savings
                    })

        # 2. Check if integer columns could use smaller types
        for col in df.select_dtypes(include=['int64']).columns:
            col_min = df[col].min()
            col_max = df[col].max()

            suggested_type = None
            if col_min >= -128 and col_max <= 127:
                suggested_type = "int8"
            elif col_min >= -32768 and col_max <= 32767:
                suggested_type = "int16"
            elif col_min >= -2147483648 and col_max <= 2147483647:
                suggested_type = "int32"

            if suggested_type:
                result["potential_optimizations"].append({
                    "column": col,
                    "current_type": "int64",
                    "suggested_type": suggested_type,
                    "range": [int(col_min), int(col_max)],
                    "estimated_savings_mb": result["memory_usage"]["by_column"][col] * 0.5  # Approximate savings
                })

        # 3. Check for high cardinality categorical columns that should be object
        for col in df.select_dtypes(include=['category']).columns:
            if col in result["unique_counts"] and result["unique_counts"][col]:
                unique_ratio = result["unique_counts"][col] / len(df)
                if unique_ratio > 0.8:  # More than 80% unique values
                    result["potential_optimizations"].append({
                        "column": col,
                        "current_type": "category",
                        "suggested_type": "object",
                        "unique_ratio": unique_ratio,
                        "reason": "High cardinality categorical"
                    })

        logger.debug(f"DataFrame analysis complete: {len(df)} rows, "
                     f"{len(df.columns)} columns, "
                     f"{result['memory_usage']['total_mb']:.2f}MB, "
                     f"{len(result['potential_optimizations'])} potential optimizations")

        return result
    except Exception as e:
        logger.error(f"Error analyzing DataFrame: {str(e)}")
        result["error"] = str(e)
        return result


def create_sample_dataframe(
        df: pd.DataFrame,
        sample_size: int = 1000,
        random_seed: int = 42,
        preserve_dtypes: bool = True,
        logger: Optional[Logger] = None
) -> pd.DataFrame:
    """
    Create a representative sample of a DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Source DataFrame
    sample_size : int
        Number of rows in the sample
    random_seed : int
        Random seed for reproducibility
    preserve_dtypes : bool
        Whether to preserve data types in the sample
    logger : Logger, optional
        Logger instance

    Returns:
    --------
    pd.DataFrame
        Sample DataFrame
    """
    # Initialize logger if not provided
    if logger is None:
        logger = custom_logging.get_logger("dataframe_helpers")

    if len(df) <= sample_size:
        logger.debug(
            f"DataFrame has {len(df)} rows, which is <= requested sample size {sample_size}. Returning full DataFrame.")
        return df.copy()

    logger.debug(f"Creating sample of {sample_size} rows from DataFrame with {len(df)} rows")

    # Create sample - use stratified sampling if possible
    try:
        # Try to identify a good column for stratification
        candidate_cols = []
        for col in df.columns:
            if df[col].dtype.name in ['category', 'object', 'bool'] or 'int' in df[col].dtype.name:
                if df[col].nunique() > 1 and df[col].nunique() <= 100:
                    candidate_cols.append(col)

        # If we have a good stratification column, use it
        if candidate_cols:
            strat_col = min(candidate_cols, key=lambda c: abs(df[c].nunique() - 10))

            # Get value counts for stratification
            value_counts = df[strat_col].value_counts(normalize=True)

            # Calculate number of samples per stratum
            samples_per_value = {k: max(1, int(v * sample_size)) for k, v in value_counts.items()}

            # Create stratified sample
            sample_dfs = []
            for val, count in samples_per_value.items():
                stratum = df[df[strat_col] == val]
                if len(stratum) >= count:
                    sample_dfs.append(stratum.sample(n=count, random_state=random_seed))
                else:
                    sample_dfs.append(stratum)  # Take all if not enough

            sample = pd.concat(sample_dfs).sample(frac=1, random_state=random_seed).reset_index(drop=True)

            # Trim to exact sample size if needed
            if len(sample) > sample_size:
                sample = sample.iloc[:sample_size]

            logger.debug(f"Created stratified sample using column '{strat_col}'")
        else:
            # Fall back to simple random sampling
            sample = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
            logger.debug("Created simple random sample")

        # Preserve data types if requested
        if preserve_dtypes:
            for col in sample.columns:
                sample[col] = sample[col].astype(df[col].dtype)

        logger.debug(f"Sample created successfully: {len(sample)} rows, {len(sample.columns)} columns")
        return sample
    except Exception as e:
        logger.warning(f"Error creating sample: {str(e)}. Falling back to simple random sample.")
        try:
            return df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)
        except Exception:
            logger.warning(f"Error with random sample. Falling back to head.")
            return df.head(sample_size).reset_index(drop=True)