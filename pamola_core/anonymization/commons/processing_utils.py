"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Anonymization Processing Utilities
Description: Common data processing utilities for anonymization operations
Author: PAMOLA Core Team
Created: 2024
License: BSD 3-Clause

This module provides common processing functions for anonymization operations,
including data chunking, generalization, and transformation utilities.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Callable, Iterator

import numpy as np
import pandas as pd

from pamola_core.utils.progress import HierarchicalProgressTracker, ProgressTracker

logger = logging.getLogger(__name__)


def process_in_chunks(
    df: pd.DataFrame,
    process_function: Callable,
    chunk_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> Union[pd.DataFrame, None, Any]:
    """
    Process a DataFrame in chunks to handle large datasets efficiently.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument
    chunk_size : int, optional
        Number of rows to process in each chunk (default: 10000)
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation
    **kwargs : dict
        Additional arguments to pass to the process_function

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame
    """
    # Check if DataFrame is empty or smaller than chunk size
    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning as is")
        return df

    if len(df) <= chunk_size:
        # If DataFrame is smaller than chunk size, process it directly
        return process_function(df, **kwargs)

    # Calculate total number of chunks
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    # Update progress if tracker is provided
    if progress_tracker:
        progress_tracker.total = total_chunks
        progress_tracker.update(
            0, {"step": "Processing in chunks", "total_chunks": total_chunks}
        )

    # Initialize result with DataFrame structure but no rows
    result = pd.DataFrame(columns=df.columns)

    # Process in chunks with logging and progress tracking
    start_time = time.time()
    processed_rows = 0

    try:
        for i in range(0, len(df), chunk_size):
            chunk_num = i // chunk_size
            chunk_start = i
            chunk_end = min(i + chunk_size, len(df))
            chunk_size = chunk_end - chunk_start

            logger.debug(
                f"Processing chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
            )

            # Extract and process chunk
            chunk = df.iloc[chunk_start:chunk_end].copy()

            try:
                processed_chunk = process_function(chunk, **kwargs)

                # Validate processed chunk
                if not isinstance(processed_chunk, pd.DataFrame):
                    logger.error(
                        f"Chunk {chunk_num + 1} processing failed: function did not return a DataFrame"
                    )
                    continue

                if len(processed_chunk) != len(chunk):
                    logger.warning(
                        f"Processed chunk {chunk_num + 1} has different row count: "
                        f"got {len(processed_chunk)}, expected {len(chunk)}"
                    )

                # Append to result
                result = pd.concat([result, processed_chunk], ignore_index=True)
                processed_rows += len(chunk)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                # Continue with next chunk for fault tolerance

            # Update progress
            if progress_tracker:
                progress_tracker.update(
                    chunk_num + 1,
                    {
                        "step": "Processing chunks",
                        "chunk": chunk_num + 1,
                        "processed_rows": processed_rows,
                        "total_rows": len(df),
                    },
                )

        logger.info(
            f"Processed {processed_rows}/{len(df)} rows in {total_chunks} chunks"
        )

    except Exception as e:
        logger.error(f"Error during chunk processing: {str(e)}")
        # Return partially processed result if any
        logger.warning(f"Returning partially processed result with {len(result)} rows")

    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk processing completed in {elapsed_time:.2f} seconds")

    return result


def get_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000
) -> Iterator[pd.DataFrame]:
    """
    Generate chunks of a DataFrame for efficient processing of large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to chunk
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000)

    Yields:
    -------
    pd.DataFrame
        Chunk of the original DataFrame
    """
    # Check if DataFrame is empty
    if len(df) == 0:
        logger.warning("Empty DataFrame provided to chunking function")
        yield df
        return

    # Calculate total number of chunks
    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.info(f"Splitting DataFrame with {len(df)} rows into {total_chunks} chunks")

    # Process in chunks
    for i in range(0, len(df), chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, len(df))
        chunk_num = i // chunk_size

        logger.debug(
            f"Yielding chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
        )
        yield df.iloc[chunk_start:chunk_end].copy()


def process_dataframe_parallel(
    df: pd.DataFrame,
    process_function: Callable,
    n_jobs: int = -1,
    chunk_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Process a DataFrame in parallel using joblib for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    process_function : Callable
        Function to apply to each chunk
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 to use all processors) (default: -1)
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000)
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation
    **kwargs : dict
        Additional arguments to pass to the process_function

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        logger.warning("joblib not installed, falling back to sequential processing")
        return process_in_chunks(
            df, process_function, chunk_size, progress_tracker, **kwargs
        )

    # Check if DataFrame is empty or smaller than chunk size
    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning as is")
        return df

    if len(df) <= chunk_size:
        # If DataFrame is smaller than chunk size, process it directly
        return process_function(df, **kwargs)

    # Split DataFrame into chunks
    chunks = [chunk for chunk in get_dataframe_chunks(df, chunk_size=chunk_size)]
    total_chunks = len(chunks)

    # Update progress if tracker is provided
    if progress_tracker:
        progress_tracker.total = total_chunks
        progress_tracker.update(
            0, {"step": "Parallel processing setup", "total_chunks": total_chunks}
        )

    logger.info(
        f"Processing {len(df)} rows in {total_chunks} chunks with {n_jobs} workers"
    )

    try:
        # Process chunks in parallel
        start_time = time.time()

        # Define processing function with progress tracking
        def process_with_progress(chunk_idx, chunk):
            try:
                result = process_function(chunk, **kwargs)
                if progress_tracker:
                    progress_tracker.update(
                        chunk_idx + 1,
                        {
                            "step": "Parallel processing",
                            "chunk": chunk_idx + 1,
                            "processed_rows": (
                                (chunk_idx + 1) * chunk_size
                                if chunk_idx < total_chunks - 1
                                else len(df)
                            ),
                            "total_rows": len(df),
                        },
                    )
                return result
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                return None

        # Execute in parallel
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(i, chunk) for i, chunk in enumerate(chunks)
        )

        # Filter out None results (failed chunks)
        processed_chunks = [chunk for chunk in processed_chunks if chunk is not None]

        # Combine results
        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
        else:
            logger.error("All chunks failed processing, returning empty DataFrame")
            result = pd.DataFrame(columns=df.columns)

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        # Compute final result
        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Parallel finalization",
                    "total_chunks": total_chunks,
                },
            )

        return result

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        logger.warning("Falling back to sequential processing")
        return process_in_chunks(
            df, process_function, chunk_size, progress_tracker, **kwargs
        )


def process_dataframe_dask(
    df: pd.DataFrame,
    process_function: Callable,
    process_function_backup: Callable,
    chunk_size: int = 10000,
    npartitions: Optional[int] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Process a DataFrame in dask using dask for large datasets.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to process
    process_function : Callable
        Function to apply to each chunk
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000)
    npartitions : Optional[int]
        Number of partitions to use with Dask (default: None)
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation
    **kwargs : dict
        Additional arguments to pass to the process_function

    Returns:
    --------
    pd.DataFrame
        The processed DataFrame
    """
    try:
        import dask.dataframe as dd
    except ImportError:
        logger.warning("dask not installed, falling back to sequential processing")
        return process_in_chunks(
            df, process_function, chunk_size, progress_tracker, **kwargs
        )

    # Check if DataFrame is empty or smaller than chunk size
    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning as is")
        return df

    # Convert to Dask DataFrame
    total_rows = len(df)
    if npartitions is None or npartitions < 1:
        nparts = (total_rows + chunk_size - 1) // chunk_size
    else:
        nparts = npartitions

    ddf = dd.from_pandas(df, npartitions=nparts)

    # Update progress if tracker is provided
    if progress_tracker:
        progress_tracker.total = nparts
        progress_tracker.update(
            1, {"step": "Dask processing setup", "total_parts": nparts}
        )

    logger.info(f"Processing {total_rows} rows in {nparts} chunks with Dask")

    try:
        # Process dask DataFrame
        start_time = time.time()

        # Update progress for Dask processing
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Dask processing",
                    "total_parts": nparts,
                },
            )

        # Execute the processing function
        result = process_function(ddf, **kwargs)

        # Compute elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Dask processing completed in {elapsed_time:.2f} seconds")
        # Combine results
        if result is not None:
            # Compute final result
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Dask finalization",
                        "total_parts": nparts,
                    },
                )
            return result
        else:
            logger.error("Dask processing failed, returning empty DataFrame")
            return pd.DataFrame(columns=df.columns)
    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        logger.warning("Falling back to sequential processing")
        return process_in_chunks(
            df,
            process_function=process_function_backup,
            chunk_size=chunk_size,
            progress_tracker=progress_tracker,
            **kwargs,
        )


def numeric_generalization_binning(
    series: pd.Series,
    bin_count: int,
    labels: Optional[List[str]] = None,
    handle_nulls: bool = True,
) -> pd.Series:
    """
    Generalize numeric values by binning them into intervals.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    bin_count : int
        Number of bins to create
    labels : List[str], optional
        Custom labels for the bins (default: None, will use interval notation)
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series
    """
    # Create a mask for null values at the beginning of the function
    null_mask = series.isnull()

    if handle_nulls:
        # Only process non-null values
        non_null_series = series[~null_mask]
    else:
        non_null_series = series

    # Check if series is empty after null handling
    if len(non_null_series) == 0:
        logger.warning("No non-null values to bin, returning original series")
        return series

    # Calculate bin edges
    try:
        min_val = non_null_series.min()
        max_val = non_null_series.max()

        # Ensure min_val and max_val are different to avoid errors in pd.cut
        if min_val == max_val:
            # If all values are the same, create a single bin
            min_val = min_val - 0.5
            max_val = max_val + 0.5

        bin_edges = np.linspace(min_val, max_val, bin_count + 1)

        # Create default labels if not provided
        if labels is None:
            labels = [
                f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(bin_count)
            ]

        # Apply binning to non-null values
        binned = pd.cut(
            non_null_series, bins=bin_edges, labels=labels, include_lowest=True
        )

        if handle_nulls:
            # Create result Series with same index as original
            result = pd.Series(index=series.index, dtype=binned.dtype)
            # Fill in binned values for non-null positions
            result[~null_mask] = binned
            # Keep nulls as nulls
            result[null_mask] = None
            return result
        else:
            return binned

    except Exception as e:
        logger.error(f"Error in numeric binning: {str(e)}")
        # Return original series on error for fault tolerance
        return series


def numeric_generalization_rounding(
    series: pd.Series, precision: int, handle_nulls: bool = True
) -> pd.Series:
    """
    Generalize numeric values by rounding to a specified precision.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    precision : int
        Number of decimal places to round to (can be negative for rounding to 10s, 100s, etc.)
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series
    """
    # Create a mask for null values at the beginning of the function
    null_mask = series.isnull()

    if handle_nulls:
        # Only process non-null values
        non_null_series = series[~null_mask]
    else:
        non_null_series = series

    # Check if series is empty after null handling
    if len(non_null_series) == 0:
        logger.warning("No non-null values to round, returning original series")
        return series

    try:
        # Apply rounding
        if precision >= 0:
            # Round to decimal places
            rounded = non_null_series.round(precision)
        else:
            # Round to nearest 10^|precision|
            factor = 10 ** abs(precision)
            rounded = (non_null_series / factor).round() * factor

        if handle_nulls:
            # Create result Series with same index as original
            result = pd.Series(index=series.index, dtype=rounded.dtype)
            # Fill in rounded values for non-null positions
            result[~null_mask] = rounded
            # Keep nulls as nulls
            result[null_mask] = None
            return result
        else:
            return rounded

    except Exception as e:
        logger.error(f"Error in numeric rounding: {str(e)}")
        # Return original series on error for fault tolerance
        return series


def numeric_generalization_range(
    series: pd.Series, range_limits: Tuple[float, float], handle_nulls: bool = True
) -> pd.Series:
    """
    Generalize numeric values by mapping to custom range intervals.

    Parameters:
    -----------
    series : pd.Series
        The numeric series to generalize
    range_limits : Tuple[float, float]
        The (min, max) range limits
    handle_nulls : bool, optional
        Whether to handle null values specially (default: True)

    Returns:
    --------
    pd.Series
        The generalized series with range labels
    """
    # Create a mask for null values at the beginning of the function
    null_mask = series.isnull()

    try:
        min_val, max_val = range_limits

        if handle_nulls:
            # Only process non-null values
            non_null_series = series[~null_mask]
        else:
            non_null_series = series

        # Check if series is empty after null handling
        if len(non_null_series) == 0:
            logger.warning("No non-null values to range-map, returning original series")
            return series

        # Calculate which values fall into the range
        in_range = (non_null_series >= min_val) & (non_null_series < max_val)

        # Create result labels
        result_values = pd.Series(index=non_null_series.index, dtype="object")
        result_values[in_range] = f"{min_val}-{max_val}"
        result_values[~in_range & (non_null_series < min_val)] = f"<{min_val}"
        result_values[~in_range & (non_null_series >= max_val)] = f">={max_val}"

        if handle_nulls:
            # Create result Series with same index as original
            result = pd.Series(index=series.index, dtype="object")
            # Fill in values for non-null positions
            result[~null_mask] = result_values
            # Keep nulls as nulls
            result[null_mask] = None
            return result
        else:
            return result_values

    except Exception as e:
        logger.error(f"Error in range mapping: {str(e)}")
        # Return original series on error for fault tolerance
        return series


def process_nulls(series: pd.Series, null_strategy: str) -> pd.Series:
    """
    Process null values according to the specified strategy.

    Parameters:
    -----------
    series : pd.Series
        The series to process
    null_strategy : str
        Strategy for handling nulls: "PRESERVE", "EXCLUDE", or "ERROR"

    Returns:
    --------
    pd.Series
        The processed series

    Raises:
    -------
    ValueError
        If null_strategy is "ERROR" and nulls are found
    """
    null_count = series.isnull().sum()

    if null_count == 0:
        # No nulls to process
        return series

    if null_strategy == "PRESERVE":
        # Keep nulls as they are
        return series

    elif null_strategy == "EXCLUDE":
        # Return only non-null values
        logger.info(f"Excluding {null_count} null values")
        return series.dropna()

    elif null_strategy == "ERROR":
        # Raise error if nulls found
        error_message = (
            f"Field contains {null_count} null values, and null_strategy is 'ERROR'"
        )
        logger.error(error_message)
        raise ValueError(error_message)

    else:
        error_message = f"Invalid null_strategy: {null_strategy}"
        logger.error(error_message)
        raise ValueError(error_message)


def generate_output_field_name(
    df: pd.DataFrame,
    field_name: str,
    mode: str,
    output_field_name: Optional[str],
    column_prefix: str,
) -> str:
    """
    Generate the appropriate output field name based on mode and parameters.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to check field names against
    field_name : str
        Original field name
    mode : str
        Operation mode ("REPLACE" or "ENRICH")
    output_field_name : str, optional
        User-specified output field name (can be None)
    column_prefix : str
        Prefix to use when creating new field name in ENRICH mode

    Returns:
    --------
    str
        The output field name to use
    """
    # Determine output field name based on mode
    if mode == "REPLACE":
        return field_name
    else:  # ENRICH mode
        if output_field_name:
            output_field = output_field_name
        else:
            output_field = f"{column_prefix}{field_name}"

        # Check if output field already exists in DataFrame
        if output_field in df.columns:
            logger.warning(
                f"Output field '{output_field}' already exists and will be overwritten"
            )

        return output_field


def prepare_output_directory(task_dir: Path, subdirectory: str) -> Path:
    """
    Prepare an output directory within the task directory.

    Parameters:
    -----------
    task_dir : Path
        The task directory
    subdirectory : str
        Name of the subdirectory to create

    Returns:
    --------
    Path
        Path to the created subdirectory
    """
    output_dir = task_dir / subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created directory: {output_dir}")
    return output_dir
