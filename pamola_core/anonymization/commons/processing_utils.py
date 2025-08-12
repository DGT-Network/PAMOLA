"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: processing_utils.py
Description: Utility functions for efficient and scalable DataFrame processing.

Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides utility functions for processing pandas DataFrames in
parallel or in chunks, supporting both single-machine and distributed workflows.

Key features:
- Parallel DataFrame processing using joblib
- Chunk-wise DataFrame processing for memory efficiency
- Progress tracking integration for long-running operations
"""

import logging
from typing import Tuple, Callable, Optional

import pandas as pd
from joblib import Parallel, delayed
from pamola_core.utils.io_helpers.dask_utils import convert_from_dask, convert_to_dask
from pamola_core.utils.ops.op_data_processing import get_dataframe_chunks
from pamola_core.utils.progress import HierarchicalProgressTracker


def process_dataframe_using_dask(
    df: pd.DataFrame,
    process_function: Callable,
    is_use_batch_dask: bool = False,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool]:
    """
    Process DataFrame using Dask.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument.
    is_use_batch_dask : bool, optional
        Whether to use batch processing with Dask (default: False).
    progress_tracker : Optional[HierarchicalProgressTracker], optional
        Progress tracker for monitoring processing status (default: None).
    task_logger : logging.Logger, optional
        Logger to use for task logging (default: None).
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    task_logger = task_logger or logging.getLogger(__name__)

    # Dask processing parameters
    npartitions = kwargs.get("npartitions", 2)
    dask_partition_size = kwargs.get("dask_partition_size", "100MB")

    # Total number rows in the DataFrame
    total_rows = len(df)
    task_logger.info(f"Process DataFrame {total_rows} rows using Dask")

    try:
        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Dask processing setup",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )
        # Convert to Dask DataFrame
        ddf, npartitions = convert_to_dask(
            df,
            dask_npartitions=npartitions,
            dask_partition_size=dask_partition_size or "100MB",
            logger=task_logger,
        )

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Dask processing setup",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )

        task_logger.info(
            f"Processing {total_rows} rows in {npartitions} chunks with Dask"
        )
        task_logger.info(
            f"Batch Dask mode: {'enabled' if is_use_batch_dask else 'disabled'}"
        )

        if is_use_batch_dask:
            # Apply the processing function to the chunk
            processed_partitions = process_function(ddf, **kwargs)
        else:
            # Define a function for processing that can be applied to Dask partitions
            def process_partition(partition):
                processed_partition = process_function(
                    partition.copy(deep=True), **kwargs
                )
                return processed_partition

            # Apply to Dask DataFrame
            processed_partitions = ddf.map_partitions(process_partition)

        task_logger.info(
            f"Processing completed for {total_rows} rows in {npartitions} partitions"
        )

        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Dask finalization",
                    "total_parts": npartitions,
                    "dask_partition_size": dask_partition_size,
                    "total_rows": total_rows,
                },
            )

        # Convert back to pandas
        result_df = convert_from_dask(processed_partitions)

        return result_df, True
    except Exception as e:
        task_logger.error(f"Error during process using Dask: {str(e)}")
        task_logger.warning("Returning as is")
        return df, False


def process_dataframe_using_joblib(
    df: pd.DataFrame,
    process_function: Callable,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool]:
    """
    Process DataFrame using Joblib.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument.
    progress_tracker : Optional[HierarchicalProgressTracker], optional
        Progress tracker for monitoring processing status (default: None).
    task_logger : Optional[logging.Logger], optional
        Logger to use for task logging (default: None).
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    task_logger = task_logger or logging.getLogger(__name__)
    task_logger.info(f"Process DataFrame {len(df)} rows using Joblib")

    # Extract parameters from kwargs
    n_jobs = kwargs.get("parallel_processes", -1)
    chunk_size = kwargs.get("chunk_size", 10000)

    if n_jobs <= 0 and n_jobs != -1:
        return df, False

    try:
        # Process with pandas in chunks
        chunks = list(get_dataframe_chunks(df, chunk_size))
        total_chunks = len(chunks)
        total_rows = len(df)

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(
                1,
                {
                    "step": "Parallel processing setup",
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                },
            )

        task_logger.info(
            f"Processing {total_rows} rows in {total_chunks} chunks with Joblib"
        )

        # Function to process each chunk with error handling
        def process_with_progress(chunk):
            try:
                processed_chunk = process_function(chunk, **kwargs)
                return processed_chunk
            except Exception as e:
                return None

        # Update progress tracker for each chunk
        if progress_tracker:
            progress_tracker.update(
                2,
                {
                    "step": "Parallel processing",
                    "total_chunks": total_chunks,
                    "total_rows": total_rows,
                },
            )

        # Directly use the generator to iterate through chunks
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(chunk) for chunk in chunks
        )

        task_logger.info(
            f"Processing completed for {total_rows} rows in {total_chunks} chunks"
        )

        # Compute final result
        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Parallel finalization",
                    "total_chunks": total_chunks,
                },
            )

        # Check if any processed chunks are None
        if any(chunk is None for chunk in processed_chunks):
            task_logger.warning("Some chunks failed to process.")
            return df, False

        return pd.concat(processed_chunks, ignore_index=True), True
    except Exception as e:
        task_logger.error(f"Error during process using Joblib: {str(e)}")
        task_logger.warning("Returning as is")
        return df, False


def process_dataframe_using_chunk(
    df: pd.DataFrame,
    process_function: Callable,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, bool]:
    """
    Process DataFrame using chunk.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument.
    progress_tracker : Optional[HierarchicalProgressTracker], optional
        Progress tracker for monitoring processing status (default: None).
    task_logger : Optional[logging.Logger], optional
        Logger to use for task logging (default: None).
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    task_logger = task_logger or logging.getLogger(__name__)
    task_logger.info(f"Process DataFrame {len(df)} rows using chunk")
    # Extract parameters from kwargs
    chunk_size = kwargs.get("chunk_size", 10000)
    if chunk_size <= 1:
        task_logger.warning("Chunk config not valid! Returning as is")
        return df, False

    task_logger.info(
        f"Process DataFrame {len(df)} rows using chunk with {chunk_size} chunk size"
    )

    processed_chunks = []
    try:
        # Process with pandas in chunks
        chunks = list(get_dataframe_chunks(df, chunk_size))
        total_chunks = len(chunks)
        # Update progress if tracker is provided
        task_logger.info(
            f"Processing {len(df)} rows in {total_chunks} chunks with chunk size {chunk_size}"
        )
        if progress_tracker:
            progress_tracker.total = total_chunks
            progress_tracker.update(
                1, {"step": "Chunk processing setup", "total_chunks": total_chunks}
            )

        for i, chunk in enumerate(chunks):
            try:
                if progress_tracker:
                    progress_tracker.update(
                        i + 1, {"step": f"Processing chunk {i + 1}/{total_chunks}"}
                    )
                # Apply the processing function to the chunk
                processed_chunk = process_function(chunk, **kwargs)

                # Accumulate the results
                processed_chunks.append(processed_chunk)
            except Exception as e:
                # Log any error encountered while processing the chunk
                task_logger.error(f"Error processing chunk {i + 1}: {str(e)}")
                processed_chunks.append(None)
                continue  # Continue with the next chunk even if an error occurs

        task_logger.info(
            f"Processing completed for {len(df)} rows in {total_chunks} chunks"
        )
        # Compute final result
        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Chunk finalization",
                    "total_chunks": total_chunks,
                },
            )
        # Check if any processed chunks are None
        if any(chunk is None for chunk in processed_chunks):
            task_logger.warning("Some chunks failed to process.")
            return df, False

        return pd.concat(processed_chunks, ignore_index=True), True
    except Exception as e:
        # Catch any error that occurs during the entire processing loop
        task_logger.error(f"Error during process using chunk: {str(e)}")
        task_logger.warning("Returning as is")
        return df, False
