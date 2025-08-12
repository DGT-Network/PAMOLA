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
import time
from typing import List, Dict, Tuple, Iterable, Iterator, Callable, Any, Union, Optional

import pandas as pd
from joblib import Parallel, delayed

from pamola_core.utils.progress import HierarchicalProgressTracker

from pamola_core.transformations.commons.aggregation_utils import (
    apply_custom_aggregations_post_dask,
    build_aggregation_dict,
    flatten_multiindex_columns,
    is_dask_compatible_function,
)
from pamola_core.transformations.commons.validation_utils import (
    validate_dataframe,
    validate_group_and_aggregation_fields,
    validate_join_type,
)

logger = logging.getLogger(__name__)


def process_dataframe_with_config(
    df: pd.DataFrame,
    process_function: Callable,
    chunk_size: int = 10000,
    use_dask: bool = False,
    npartitions: int = 1,
    meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
    use_vectorization: bool = False,
    parallel_processes: int = 1,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
    **kwargs,
) -> Union[pd.DataFrame, None, Any]:
    """
    Process a DataFrame to handle large datasets efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument.
    chunk_size : int, optional
        Number of rows to process in each chunk (default: 10000).
    use_dask : bool, optional
        Whether to use Dask for processing (default: False).
    npartitions : int, optional
        Number of partitions use with Dask (default: 1).
    meta : Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple], optional
        Meta of output use with Dask.
    use_vectorization : bool, optional
        Whether to use vectorized (parallel) processing (default: False).
    parallel_processes : int, optional
        Number of processes use with vectorized (parallel) (default: 1).
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation.
    task_logger : Optional[logging.Logger]
        Logger for tracking task progress and debugging.
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame.
    """
    # Initialize task logger
    if task_logger:
        logger = task_logger
        
    if len(df) == 0:
        logger.warning("Empty DataFrame provided! Returning as is")
        return df

    if len(df) <= chunk_size:
        logger.warning("Small DataFrame! Process as usual")
        return process_function(df, **kwargs)

    processed_df = None
    flag_processed = False

    logger.info("Process with config")

    if not flag_processed and use_dask:
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Dask")
        logger.info(f"Parallel Workers: {npartitions}")
        logger.info(f"Using dask processing with chunk size {chunk_size}")
        if progress_tracker:
            progress_tracker.update(0, {"step": "Setting up dask processing"})

        logger.info("Process using Dask")

        processed_df, flag_processed = _process_dataframe_using_dask(
            df=df,
            process_function=process_function,
            npartitions=npartitions,
            chunksize=chunk_size,
            progress_tracker=progress_tracker,
            meta=meta,
            **kwargs,
        )

        if flag_processed:
            logger.info("Completed using Dask")

    if not flag_processed and use_vectorization:
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Joblib")
        logger.info(f"Parallel Workers: {parallel_processes}")
        logger.info(f"Using vectorized processing with chunk size {chunk_size}")
        if progress_tracker:
            progress_tracker.update(0, {"step": "Setting up vectorized processing"})

        logger.info("Process using Joblib")

        processed_df, flag_processed = _process_dataframe_using_joblib(
            df=df,
            process_function=process_function,
            n_jobs=parallel_processes,
            chunk_size=chunk_size,
            progress_tracker=progress_tracker,
            **kwargs,
        )

        if flag_processed:
            logger.info("Completed using Joblib")

    if not flag_processed and chunk_size > 1:
        # Regular chunk processing
        logger.info(f"Processing in chunks with chunk size {chunk_size}")
        total_chunks = (len(df) + chunk_size - 1) // chunk_size
        logger.info(f"Total chunks to process: {total_chunks}")
        if progress_tracker:
            progress_tracker.update(
                0, {"step": "Processing in chunks", "total_chunks": total_chunks}
            )

        logger.info("Process using chunk")

        processed_df, flag_processed = _process_dataframe_using_chunk(
            df=df,
            process_function=process_function,
            chunk_size=chunk_size,
            progress_tracker=progress_tracker,
            **kwargs,
        )

        if flag_processed:
            logger.info("Completed using chunk")

    if not flag_processed:
        logger.info("Fallback process as usual")

        processed_df = process_function(df, **kwargs)
        flag_processed = True

    return processed_df


def _generate_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000
) -> Iterator[tuple]:
    """
    Generate chunks of a DataFrame for efficient processing of large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to chunk.
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000).

    Yields
    ------
    tuple
        Chunk of the original DataFrame with information.
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided to chunking function")
        yield df, 0, 0, 0, 0
        return

    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.info(
        f"Splitting DataFrame with {len(df)} rows into {total_chunks} chunks with size {chunk_size}"
    )

    for i in range(0, len(df), chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, len(df))
        chunk_num = i // chunk_size

        logger.debug(
            f"Yielding chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
        )
        yield df.iloc[chunk_start:chunk_end].copy(
            deep=True
        ), chunk_num, chunk_start, chunk_end, total_chunks


def _process_dataframe_using_dask(
    df: pd.DataFrame,
    process_function: Callable,
    npartitions: int = 2,
    chunksize: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
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
    npartitions : int, optional
        Number of partitions use with Dask (default: 2).
    chunksize : int, optional
        Number of rows to process in each chunk (default: 10000).
    meta : Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple], optional
        Meta of output use with Dask.
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    import dask.dataframe as dd

    if npartitions <= 1 and chunksize <= 0:
        logger.warning("Dask config not valid! Returning as is")
        return df, False

    logger.info(
        f"Process DataFrame {len(df)} rows using Dask with {npartitions} partitions, {chunksize} chunk size"
    )

    try:
        # Convert to Dask DataFrame
        total_rows = len(df)
        if npartitions is None or npartitions < 1:
            nparts = (total_rows + chunksize - 1) // chunksize
        else:
            nparts = npartitions

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.total = nparts
            progress_tracker.update(
                1, {"step": "Dask processing setup", "total_parts": nparts}
            )

        ddf = dd.from_pandas(df, npartitions=nparts)

        # Update progress if tracker is provided
        if progress_tracker:
            progress_tracker.update(
                2, {"step": "Dask processing setup", "total_parts": nparts}
            )

        logger.info(f"Processing {total_rows} rows in {nparts} chunks with Dask")

        # Define a function for processing that can be applied to Dask partitions
        def process_partition(partition):
            processed_partition = process_function(partition.copy(deep=True), **kwargs)
            return processed_partition

        # Apply to Dask DataFrame
        processed_partitions = ddf.map_partitions(process_partition)
        if progress_tracker:
            progress_tracker.update(
                3,
                {
                    "step": "Dask finalization",
                    "total_parts": nparts,
                },
            )
        return processed_partitions.compute(), True
    except Exception as e:
        logger.error(f"Error during process using Dask: {str(e)}")
        logger.warning("Returning as is")
        return df, False


def _process_dataframe_using_joblib(
    df: pd.DataFrame,
    process_function: Callable,
    n_jobs: int = -1,
    chunk_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
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
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 to use all processors) (default: -1).
    chunk_size : int, optional
        Number of rows to process in each chunk (default: 10000).
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    from joblib import Parallel, delayed

    if n_jobs <= 0 and n_jobs != -1:
        logger.warning("Joblib config not valid! Returning as is")
        return df, False

    logger.info(
        f"Process DataFrame {len(df)} rows using Joblib with {n_jobs} workers, {chunk_size} chunk size"
    )

    try:
        # Update progress if tracker is provided
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        if progress_tracker:
            progress_tracker.total = total_chunks
            progress_tracker.update(
                1, {"step": "Parallel processing setup", "total_chunks": total_chunks}
            )

        # Function to process each chunk with error handling
        def process_with_progress(chunk, chunk_idx):
            try:
                processed_chunk = process_function(chunk, **kwargs)
                if progress_tracker:
                    progress_tracker.update(
                        chunk_idx + 1,
                        {
                            "step": "Parallel processing",
                            "chunk": chunk_idx + 1,
                            "processed_rows": (
                                (chunk_idx + 1) * chunk_size
                                if chunk_idx < total_chunks - 1
                                else total_rows
                            ),
                            "total_rows": total_rows,
                        },
                    )
                return processed_chunk
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                return None

        # Directly use the generator to iterate through chunks
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(chunk, idx)
            for idx, (
                chunk,
                chunk_num,
                chunk_start,
                chunk_end,
                total_chunks,
            ) in enumerate(_generate_dataframe_chunks(df, chunk_size=chunk_size))
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
            logger.warning("Some chunks failed to process.")
            return df, False

        return pd.concat(processed_chunks, ignore_index=True), True
    except Exception as e:
        logger.error(f"Error during process using Joblib: {str(e)}")
        logger.warning("Returning as is")
        return df, False


def _process_dataframe_using_chunk(
    df: pd.DataFrame,
    process_function: Callable,
    chunk_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
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
    chunk_size : int, optional
        Number of rows to process in each chunk (default: 10000).
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    Tuple[pd.DataFrame, bool]
        The processed DataFrame.
    """
    if chunk_size <= 1:
        logger.warning("Chunk config not valid! Returning as is")
        return df, False

    logger.info(
        f"Process DataFrame {len(df)} rows using chunk with {chunk_size} chunk size"
    )

    processed_chunks = []
    try:
        # Update progress if tracker is provided
        total_rows = len(df)
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        if progress_tracker:
            progress_tracker.total = total_chunks
            progress_tracker.update(
                1, {"step": "Chunk processing setup", "total_chunks": total_chunks}
            )

        # Iterate through chunks using the generator function
        for (
            chunk,
            chunk_num,
            chunk_start,
            chunk_end,
            total_chunks,
        ) in _generate_dataframe_chunks(df, chunk_size=chunk_size):
            logger.debug(
                f"Process chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
            )

            try:
                # Apply the processing function to the chunk
                processed_chunk = process_function(chunk, **kwargs)
                if progress_tracker:
                    progress_tracker.update(
                        chunk_num + 1,
                        {
                            "step": "Parallel processing",
                            "chunk": chunk_num + 1,
                            "processed_rows": (
                                (chunk_num + 1) * chunk_size
                                if chunk_num < total_chunks - 1
                                else total_rows
                            ),
                            "total_rows": total_rows,
                        },
                    )
                # Accumulate the results
                processed_chunks.append(processed_chunk)
            except Exception as e:
                # Log any error encountered while processing the chunk
                logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                processed_chunks.append(None)
                continue  # Continue with the next chunk even if an error occurs

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
            logger.warning("Some chunks failed to process.")
            return df, False

        return pd.concat(processed_chunks, ignore_index=True), True
    except Exception as e:
        # Catch any error that occurs during the entire processing loop
        logger.error(f"Error during process using chunk: {str(e)}")
        logger.warning("Returning as is")
        return df, False


def process_in_chunks(
    df: pd.DataFrame,
    process_function: Callable,
    batch_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> Union[pd.DataFrame, None, Any]:
    """
    Process a DataFrame in chunks to handle large datasets efficiently.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk, should take a DataFrame chunk as the first argument.
    batch_size : int, optional
        Number of rows to process in each chunk (default: 10000).
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation.
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame.
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning as is")
        return df

    if len(df) <= batch_size:
        return process_function(df, **kwargs)

    total_chunks = (len(df) + batch_size - 1) // batch_size

    if progress_tracker:
        progress_tracker.total = total_chunks
        progress_tracker.update(
            0, {"step": "Processing in chunks", "total_chunks": total_chunks}
        )

    # Initialize result with DataFrame structure but no rows
    result = df.iloc[0:0].copy()
    processed_rows = 0
    start_time = time.time()

    try:
        for i in range(0, len(df), batch_size):
            chunk_num = i // batch_size
            chunk_start = i
            chunk_end = min(i + batch_size, len(df))
            chunk = df.iloc[chunk_start:chunk_end].copy(deep=True)

            logger.debug(
                f"Processing chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
            )

            try:
                processed_chunk = process_function(chunk, **kwargs)
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

                result = pd.concat([result, processed_chunk], ignore_index=True)
                processed_rows += len(chunk)

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                # Continue with next chunk for fault tolerance

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
        logger.warning(f"Returning partially processed result with {len(result)} rows")

    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Chunk processing completed in {elapsed_time:.2f} seconds")

    return result


def _get_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000
) -> Iterator[pd.DataFrame]:
    """
    Generate chunks of a DataFrame for efficient processing of large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to chunk.
    chunk_size : int, optional
        Number of rows in each chunk (default: 10000).

    Yields
    ------
    pd.DataFrame
        Chunk of the original DataFrame.
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided to chunking function")
        yield df
        return

    total_chunks = (len(df) + chunk_size - 1) // chunk_size
    logger.info(f"Splitting DataFrame with {len(df)} rows into {total_chunks} chunks")

    for i in range(0, len(df), chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, len(df))
        chunk_num = i // chunk_size
        logger.debug(
            f"Yielding chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
        )
        yield df.iloc[chunk_start:chunk_end].copy(deep=True)


def process_dataframe_parallel(
    df: pd.DataFrame,
    process_function: Callable,
    n_jobs: int = -1,
    batch_size: int = 10000,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Process a DataFrame in parallel using joblib for large datasets.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    process_function : Callable
        Function to apply to each chunk.
    n_jobs : int, optional
        Number of jobs to run in parallel (-1 to use all processors) (default: -1).
    batch_size : int, optional
        Number of rows in each chunk (default: 10000).
    progress_tracker : Optional[HierarchicalProgressTracker]
        Progress tracker for monitoring the operation.
    **kwargs : dict
        Additional arguments to pass to the process_function.

    Returns
    -------
    pd.DataFrame
        The processed DataFrame.
    """
    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning as is")
        return df

    if len(df) <= batch_size:
        return process_function(df, **kwargs)

    chunks = [chunk for chunk in _get_dataframe_chunks(df, chunk_size=batch_size)]
    total_chunks = len(chunks)

    if progress_tracker:
        progress_tracker.total = total_chunks
        progress_tracker.update(
            0, {"step": "Parallel processing setup", "total_chunks": total_chunks}
        )

    logger.info(
        f"Processing {len(df)} rows in {total_chunks} chunks with {n_jobs} workers"
    )

    start_time = time.time()

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
                            (chunk_idx + 1) * batch_size
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

    try:
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(i, chunk) for i, chunk in enumerate(chunks)
        )
        processed_chunks = [chunk for chunk in processed_chunks if chunk is not None]

        if processed_chunks:
            result = pd.concat(processed_chunks, ignore_index=True)
        else:
            logger.error("All chunks failed processing, returning empty DataFrame")
            result = pd.DataFrame(columns=df.columns)

        elapsed_time = time.time() - start_time
        logger.info(f"Parallel processing completed in {elapsed_time:.2f} seconds")
        return result

    except Exception as e:
        logger.error(f"Error during parallel processing: {str(e)}")
        logger.warning("Falling back to sequential processing")
        return process_in_chunks(
            df, process_function, batch_size, progress_tracker, **kwargs
        )


def split_dataframe(
    df: pd.DataFrame,
    field_groups: Dict[str, List[str]],
    id_field: str,
    include_id_field: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Split DataFrame into multiple DataFrames by field groups.

    Parameters
    ----------
    df : pd.DataFrame
        The source DataFrame to split.
    field_groups : Dict[str, List[str]]
        Dictionary with group names as keys and lists of field names as values.
    id_field : str
        The identifier field to use for each resulting DataFrame.
    include_id_field : bool, optional
        Whether to include the id_field in each resulting DataFrame (default: True).

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with group names as keys and split DataFrames as values.

    Raises
    ------
    ValueError
        If id_field is not in DataFrame or field_groups contains invalid fields.
    """
    result = {}

    if len(df) == 0:
        logger.warning("Empty DataFrame provided, returning empty result")
        return result

    validate_dataframe(df, [id_field])

    invalid_fields = [
        field
        for group_name, fields in field_groups.items()
        for field in fields
        if field != id_field and field not in df.columns
    ]
    if invalid_fields:
        raise ValueError(f"Invalid fields in field_groups: {invalid_fields}")

    start_time = time.time()
    logger.info(f"Splitting DataFrame into {len(field_groups)} groups")

    for group_name, fields in field_groups.items():
        fields_to_select = (
            [id_field] + fields
            if include_id_field and id_field not in fields
            else fields
        )
        valid_fields = [f for f in fields_to_select if f in df.columns]

        if valid_fields:
            result[group_name] = df[valid_fields].copy(deep=True)
            logger.debug(
                f"Group '{group_name}' created with {len(valid_fields)} columns"
            )
        else:
            logger.warning(f"Group '{group_name}' has no valid fields, skipping")

    elapsed_time = time.time() - start_time
    logger.info(
        f"Splitting completed in {elapsed_time:.4f} seconds, created {len(result)} groups"
    )

    return result


def merge_dataframes(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_key: str,
    right_key: Optional[str] = None,
    join_type: str = "left",
    suffixes: Tuple[str, str] = ("_x", "_y"),
    left_index: bool = False,
    right_index: bool = False,
    chunk_size: int = 10000,
    use_dask: bool = False,
    npartitions: Optional[int] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Merge two DataFrames with proper error handling and optional Dask support.

    Parameters
    ----------
    left_df : pd.DataFrame
        The left-side DataFrame to merge.
    right_df : pd.DataFrame
        The right-side DataFrame to merge.
    left_key : str
        Column name in the left DataFrame to join on.
    right_key : Optional[str], default=None
        Column name in the right DataFrame to join on.
        If None, defaults to the value of `left_key`.
    join_type : str, default="left"
        Type of join to perform. Options include: "left", "right", "outer", "inner".
    suffixes : Tuple[str, str], default=("_x", "_y")
        Suffixes to apply to overlapping column names in the left and right DataFrames.
    left_index : bool, default=False
        If True, use the index from the left DataFrame as the join key.
    right_index : bool, default=False
        If True, use the index from the right DataFrame as the join key.
    use_dask : bool, default=False
        If True, perform the merge using Dask for parallel/distributed execution.
    npartitions : Optional[int], default=None
        Number of Dask partitions to use. If None, will auto-determine based on data size.
    progress_tracker : Optional[HierarchicalProgressTracker], default=None
        Progress tracker for monitoring the merge operation.
    task_logger : Optional[logging.Logger], default=None
        Logger for tracking task progress and debugging.

    Returns
    -------
    pd.DataFrame
        A merged DataFrame based on the specified keys and join strategy.

    Raises
    ------
    Exception
        If the merge operation fails due to invalid keys or data incompatibility.
    """
    # Initialize task logger
    if task_logger:
        logger = task_logger

    def _resolve_keys():
        return right_key if right_key is not None else left_key

    def _do_merge() -> pd.DataFrame:
        if use_dask:
            try:
                import dask.dataframe as dd
            except ImportError:
                raise ImportError(
                    "Dask is required for distributed processing but not installed. "
                    "Install with: pip install dask[dataframe]"
                )

            left_parts = _determine_partitions(left_df, chunk_size, npartitions)
            right_parts = _determine_partitions(right_df, chunk_size, npartitions)
            d_left = dd.from_pandas(left_df, npartitions=left_parts)
            d_right = dd.from_pandas(right_df, npartitions=right_parts)
            
            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Dask")
            logger.info(f"Parallel Workers: left_parts {left_parts}")
            logger.info(f"Parallel Workers: right_parts {right_parts}")
            logger.info(f"Using dask merging datasets processing with chunk size {chunk_size}")
            # Update progress for Dask processing
            if progress_tracker:
                progress_tracker.update(
                    2,
                    {
                        "step": "Dask merging processing",
                        "left_parts": left_parts,
                        "right_parts": right_parts,
                        "chunk_size": chunk_size,
                    },
                )

            # Perform the merge operation
            ddf_result = d_left.merge(
                d_right,
                left_on=left_key,
                right_on=resolved_right_key,
                how=join_type,
                suffixes=suffixes,
                left_index=left_index,
                right_index=right_index,
            )

            # Compute elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Dask merging processing completed in {elapsed_time:.2f} seconds")
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Dask merging finalization",
                        "left_parts": left_parts,
                        "right_parts": right_parts,
                        "chunk_size": chunk_size,
                    },
                )
            return ddf_result.compute()
        else:
            logger.info(f"Using pandas to merge datasets processing!!!")
            logger.info("Parallel Disabled")
            logger.info("Parallel Engine: None")
            logger.info(f"Parallel Workers: {npartitions}")
            logger.info(f"Using pandas merging datasets processing with chunk size {chunk_size}")
            # Update progress for Dask processing
            if progress_tracker:
                progress_tracker.update(
                    2,
                    {
                        "step": "Pandas processing",
                        "left_key": left_key,
                        "right_key": right_key,
                        "chunk_size": chunk_size,
                    },
                )

            # Perform the merge operation
            df_result = pd.merge(
                left_df,
                right_df,
                left_on=left_key,
                right_on=resolved_right_key,
                how=join_type,
                suffixes=suffixes,
                left_index=left_index,
                right_index=right_index,
            )

            # Compute elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Pandas merging processing completed in {elapsed_time:.2f} seconds")
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Pandas merging finalization",
                        "left_key": left_key,
                        "right_key": right_key,
                        "chunk_size": chunk_size,
                    },
                )

            return df_result

    resolved_right_key = _resolve_keys()

    validate_dataframe(left_df, [left_key])
    validate_dataframe(right_df, [resolved_right_key])
    validate_join_type(join_type)

    logger.info(
        f"Merging DataFrames with {join_type} join on keys '{left_key}' and '{resolved_right_key}'"
    )

    if progress_tracker:
        progress_tracker.total = 3  # Setup, Processing, Finalization
        progress_tracker.update(1, {"step": "Setting up merge datasets processing"})

    start_time = time.time()
    try:
        result = _do_merge()
        return result
    except Exception as e:
        logger.error(f"Error during DataFrame merge: {e}")
        raise


def aggregate_dataframe(
    df: pd.DataFrame,
    group_by_fields: List[str],
    aggregations: Optional[Dict[str, List[str]]] = None,
    custom_aggregations: Optional[Dict[str, Callable]] = None,
    chunk_size: int = 10000,
    use_dask: bool = False,
    npartitions: Optional[int] = None,
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    Aggregate DataFrame by grouping fields, supporting both pandas and Dask.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame to be aggregated.
    group_by_fields : List[str]
        List of field names to group by.
    aggregations : Optional[Dict[str, List[str]]], optional
        Dictionary mapping column names to a list of aggregation function names (e.g., "mean", "sum").
    custom_aggregations : Optional[Dict[str, Callable]], optional
        Dictionary mapping column names to a list of custom aggregation functions (e.g., lambda, numpy functions).
    chunk_size : int, optional
        Chunk size for processing large DataFrames (default is 10000).
    use_dask : bool, optional
        Whether to use Dask instead of pandas for processing (default is False).
    npartitions : Optional[int], optional
        Number of partitions to use for Dask DataFrame if applicable (default is None).
    progress_tracker: Optional[HierarchicalProgressTracker] = None,
    task_logger : Optional[logging.Logger], default=None
        Logger for tracking task progress and debugging.

    Returns
    -------
    pd.DataFrame
        The aggregated DataFrame with flattened column names if necessary.
    """
    # Initialize task logger
    if task_logger:
        logger = task_logger

    validate_group_and_aggregation_fields(
        df, group_by_fields, aggregations, custom_aggregations
    )
    start_time = time.time()
    logger.info(f"Aggregating DataFrame by {group_by_fields})")

    # Initialize progress tracker
    if progress_tracker:
        progress_tracker.total = 3  # Setup, Processing, Finalization
        progress_tracker.update(1, {"step": "Setting up aggregate records processing"})

    agg_dict = build_aggregation_dict(aggregations, custom_aggregations)

    try:
        if use_dask:
            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Dask")
            logger.info(f"Parallel Workers: {npartitions}")
            try:
                import dask.dataframe as dd
            except ImportError:
                raise ImportError(
                    "Dask is required for distributed processing but not installed. "
                    "Install with: pip install dask[dataframe]"
                )
            
            logger.info("Parallel Enabled")
            logger.info("Parallel Engine: Dask")
            logger.info(f"Parallel Workers: {npartitions}")
            logger.info(f"Using dask to aggregate records processing with chunk size {chunk_size}")
            # Warn about custom aggregations with Dask
            if custom_aggregations:
                logger.warning(
                    "Custom aggregations may not work properly with Dask. "
                    "Consider using pandas for custom aggregations.",
                    UserWarning,
                )

            # Separate Dask-compatible and custom aggregations
            safe_agg_dict = {}
            custom_agg_dict = {}
            for col, funcs in agg_dict.items():
                safe_funcs = []
                custom_funcs = []
                for func in funcs:
                    if is_dask_compatible_function(func):
                        safe_funcs.append(func)
                    else:
                        custom_funcs.append(func)
                if safe_funcs:
                    safe_agg_dict[col] = safe_funcs
                if custom_funcs:
                    custom_agg_dict[col] = custom_funcs

            # Convert to Dask DataFrame if needed
            if not isinstance(df, dd.DataFrame):
                nparts = _determine_partitions(df, chunk_size, npartitions)
                # Update progress for Dask processing
                if progress_tracker:
                    progress_tracker.update(
                        2,
                        {
                            "step": "Dask aggregation processing",
                            "nparts": nparts,
                            "chunk_size": chunk_size,
                        },
                    )
                dask_df = dd.from_pandas(df, npartitions=nparts)
            else:
                dask_df = df

            # Perform Dask aggregation
            if safe_agg_dict:
                result = (
                    dask_df.groupby(group_by_fields)
                    .agg(safe_agg_dict)
                    .reset_index()
                    .compute()
                )
            else:
                logger.warning(
                    "No Dask-compatible aggregations found, falling back to pandas"
                )
                result = df.groupby(group_by_fields).agg(agg_dict).reset_index()

            # Apply custom aggregations on original DataFrame if needed
            if custom_agg_dict:
                result = apply_custom_aggregations_post_dask(
                    original_df=df,
                    result_df=result,
                    custom_agg_dict=custom_agg_dict,
                    group_by_fields=group_by_fields,
                )

            # Compute elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Dask aggregation processing completed in {elapsed_time:.2f} seconds")
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Dask aggregation finalization",
                        "nparts": nparts,
                        "chunk_size": chunk_size,
                    },
                )
        else:
            logger.info(f"Using pandas to aggregation records processing!!!")
            logger.info("Parallel Disabled")
            logger.info("Parallel Engine: None")
            logger.info(f"Parallel Workers: {npartitions}")
            logger.info(f"Using pandas aggregation records processing with chunk size {chunk_size}")
            # Update progress for Dask processing
            if progress_tracker:
                progress_tracker.update(
                    2,
                    {
                        "step": "Pandas aggregation processing",
                        "group_by_fields": group_by_fields,
                        "chunk_size": chunk_size,
                    },
                )

            # Perform aggregation
            result = df.groupby(group_by_fields).agg(agg_dict).reset_index()

            # Compute elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Pandas aggregation processing completed in {elapsed_time:.2f} seconds")
            if progress_tracker:
                progress_tracker.update(
                    3,
                    {
                        "step": "Pandas aggregation finalization",
                        "group_by_fields": group_by_fields,
                        "chunk_size": chunk_size,
                    },
                )

        if isinstance(result.columns, pd.MultiIndex):
            result.columns = flatten_multiindex_columns(result.columns)

        return result

    except Exception as e:
        logger.exception("Error during aggregation")
        raise


def _determine_partitions(
    df: pd.DataFrame, chunk_size: int = 10000, npartitions: Optional[int] = None
) -> int:
    """
    Determine the number of partitions for Dask DataFrame.

    Args:
        df: The input DataFrame.
        chunk_size: The size of each chunk (default is 10000).
        npartitions: Optional number of partitions.

    Returns:
        int: The number of partitions.
    """
    # Convert to Dask DataFrame
    total_rows = len(df)
    if npartitions is None or npartitions < 1:
        nparts = (total_rows + chunk_size - 1) // chunk_size
    else:
        nparts = npartitions
    return nparts
