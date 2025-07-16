
import logging
from typing import Dict, Iterator, Tuple, Iterable, Callable, Any, Union, Optional

import pandas as pd

from pamola_core.utils.progress import HierarchicalProgressTracker


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
        **kwargs
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
        logger.info("Process using Dask")
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Dask")
        logger.info(f"Parallel Workers: {npartitions}")

        processed_df, flag_processed = _process_dataframe_using_dask(
            df=df,
            process_function=process_function,
            npartitions=npartitions,
            chunksize=chunk_size,
            meta=meta,
            ** kwargs
        )

    if not flag_processed and use_vectorization:
        logger.info("Process using Joblib")
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Joblib")
        logger.info(f"Parallel Workers: {parallel_processes}")

        processed_df, flag_processed = _process_dataframe_using_joblib(
            df=df,
            process_function=process_function,
            n_jobs=parallel_processes,
            chunk_size=chunk_size,
            ** kwargs
        )

    if not flag_processed and chunk_size > 1:
        logger.info("Process using chunk")
        logger.info("Parallel Enabled")
        logger.info("Parallel Engine: Chunk")
        logger.info(f"Parallel Workers")

        processed_df, flag_processed = _process_dataframe_using_chunk(
            df=df,
            process_function=process_function,
            chunk_size=chunk_size,
            ** kwargs
        )

    if not flag_processed:
        logger.info("Fallback process as usual")

        processed_df = process_function(df, **kwargs)
        flag_processed = True

    return processed_df


def _process_dataframe_using_chunk(
        df: pd.DataFrame,
        process_function: Callable,
        chunk_size: int = 10000,
        **kwargs
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
        # Iterate through chunks using the generator function
        for chunk, chunk_num, chunk_start, chunk_end, total_chunks in _generate_dataframe_chunks(
                df, chunk_size=chunk_size
        ):
            logger.debug(f"Process chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})")

            try:
                # Apply the processing function to the chunk
                processed_chunk = process_function(chunk, **kwargs)

                # Accumulate the results
                processed_chunks.append(processed_chunk)
            except Exception as e:
                # Log any error encountered while processing the chunk
                logger.error(f"Error processing chunk {chunk_num + 1}: {str(e)}")
                processed_chunks.append(None)
                continue  # Continue with the next chunk even if an error occurs

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


def _generate_dataframe_chunks(
        df: pd.DataFrame,
        chunk_size: int = 10000
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
    logger.info(f"Splitting DataFrame with {len(df)} rows into {total_chunks} chunks with size {chunk_size}")

    for i in range(0, len(df), chunk_size):
        chunk_start = i
        chunk_end = min(i + chunk_size, len(df))
        chunk_num = i // chunk_size

        logger.debug(
            f"Yielding chunk {chunk_num + 1}/{total_chunks} (rows {chunk_start}-{chunk_end - 1})"
        )
        yield df.iloc[chunk_start:chunk_end].copy(deep=True), chunk_num, chunk_start, chunk_end, total_chunks


def _process_dataframe_using_joblib(
        df: pd.DataFrame,
        process_function: Callable,
        n_jobs: int = -1,
        chunk_size: int = 10000,
        **kwargs
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
        # Function to process each chunk with error handling
        def process_with_progress(chunk, chunk_idx):
            try:
                processed_chunk = process_function(chunk, **kwargs)
                return processed_chunk
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
                return None

        # Directly use the generator to iterate through chunks
        processed_chunks = Parallel(n_jobs=n_jobs)(
            delayed(process_with_progress)(chunk, idx)
            for idx, (chunk, chunk_num, chunk_start, chunk_end, total_chunks)
            in enumerate(_generate_dataframe_chunks(df, chunk_size=chunk_size))
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


def _process_dataframe_using_dask(
        df: pd.DataFrame,
        process_function: Callable,
        npartitions: int = 2,
        chunksize: int = 10000,
        meta: Optional[Union[pd.DataFrame, pd.Series, Dict, Iterable, Tuple]] = None,
        **kwargs
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
        if npartitions > 1:
            ddf = dd.from_pandas(df, npartitions=npartitions)
        else:
            ddf = dd.from_pandas(df, chunksize=chunksize)

        # Define a function for processing that can be applied to Dask partitions
        def process_partition(partition):
            processed_partition = process_function(partition.copy(deep=True), **kwargs)
            return processed_partition

        # Apply to Dask DataFrame
        processed_partitions = ddf.map_partitions(process_partition)

        return processed_partitions.compute(), True
    except Exception as e:
        logger.error(f"Error during process using Dask: {str(e)}")
        logger.warning("Returning as is")
        return df, False
