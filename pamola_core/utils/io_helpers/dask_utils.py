"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Dask Integration Utilities
Package:       pamola_core.utils.io_helpers
Version:       1.1.0+refactor.2025.05.22
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
   Utilities for leveraging Dask distributed computing with PAMOLA I/O modules.
   Provides seamless integration between PAMOLA's I/O operations and Dask's
   distributed computing capabilities for handling large-scale datasets.

Key Features:
   - Chunked CSV reading and writing using Dask for large-scale datasets
   - Partition-aware progress tracking with integration into the PAMOLA progress system
   - Conditional Dask usage to avoid hard dependencies if Dask is not installed
   - Computation of Dask-specific dataset statistics
   - Support for compression in CSV writing operations (Dask 2.0+)

Framework:
   This module is part of PAMOLA.CORE's I/O helper utilities and integrates
   with the core I/O module to provide distributed computing capabilities
   when processing large datasets that exceed memory constraints.

Changelog:
   1.1.0 (2025-05-22): Added compression parameter support for CSV writing
   1.0.0 (2025-01-01): Initial release with basic Dask integration
"""

from pathlib import Path
from typing import Iterator, Union, Dict, Any, Optional

import pandas as pd
import dask.dataframe as dd

from pamola_core.utils import logging
from pamola_core.utils import progress
from pamola_core.utils.ops.op_data_processing import get_memory_usage

# Configure module logger
logger = logging.get_logger("pamola_core.utils.io_helpers.dask_utils")


def is_dask_available() -> bool:
    """
    Check if Dask is available in the current environment.

    Returns:
    --------
    bool
        True if Dask is available, False otherwise
    """
    try:
        import dask.dataframe
        import dask

        # Log Dask version for debugging
        dask_version = getattr(dask, "__version__", "unknown")
        logger.debug(f"Dask is available, version: {dask_version}")
        return True
    except ImportError:
        logger.debug("Dask is not available in the current environment")
        return False


def read_csv_in_chunks(
    file_path: Union[str, Path],
    chunk_size: int,
    encoding: str,
    delimiter: str,
    quotechar: str,
    show_progress: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Read a CSV file using Dask and yield chunks as pandas DataFrames.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    chunk_size : int
        Size of each chunk to return
    encoding : str
        File encoding
    delimiter : str
        Field delimiter character
    quotechar : str
        Text qualifier character
    show_progress : bool
        Whether to display a progress bar

    Yields:
    -------
    pd.DataFrame
        Chunks of the CSV file as pandas DataFrames
    """
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        file_path = Path(file_path)
        logger.info(f"Starting Dask chunked read of {file_path}")

        # Read with Dask
        ddf = dd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            quotechar=quotechar,
            blocksize=chunk_size * 100,  # Adjust blocksize for Dask partitioning
        )

        # Get total number of partitions
        n_partitions = ddf.npartitions
        logger.debug(f"Dask created {n_partitions} partitions for the file")

        # Show progress if requested
        progress_bar = None
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=n_partitions,
                description=f"Reading {file_path.name} with Dask",
                unit="partitions",
            )

        # Process in chunks
        chunks_yielded = 0
        for i in range(0, len(ddf), chunk_size):
            end = min(i + chunk_size, len(ddf))

            # Update progress
            if progress_bar:
                progress_bar.update(1)

            # Compute chunk
            chunk = ddf.iloc[i:end].compute()
            chunks_yielded += 1

            logger.debug(f"Read Dask chunk {chunks_yielded}: {len(chunk)} rows")
            yield chunk

        # Close progress bar
        if progress_bar:
            progress_bar.close()

        logger.info(f"Completed Dask chunked read: {chunks_yielded} chunks yielded")

    except Exception as e:
        logger.error(f"Error in Dask CSV reading: {e}")
        raise


def read_full_csv(
    file_path: Union[str, Path],
    encoding: str,
    delimiter: str,
    quotechar: str,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Read a full CSV file into a pandas DataFrame using Dask.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding
    delimiter : str
        Field delimiter character
    quotechar : str
        Text qualifier character
    show_progress : bool
        Whether to display a progress bar

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the file data
    """
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        file_path = Path(file_path)
        logger.info(f"Starting Dask full read of {file_path}")

        # Read with Dask
        ddf = dd.read_csv(
            file_path, encoding=encoding, sep=delimiter, quotechar=quotechar
        )

        # Log partition information
        logger.debug(f"Dask created {ddf.npartitions} partitions for the file")

        # Compute to pandas DataFrame with progress if requested
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=100,  # Percentage-based progress
                description=f"Reading {file_path.name} with Dask",
                unit="%",
            )

            # Show initial progress
            progress_bar.update(10)

            # Compute with Dask's ProgressBar
            with ProgressBar():
                df = ddf.compute()

            # Complete progress bar
            progress_bar.update(90)
            progress_bar.close()
        else:
            # Compute without progress display
            df = ddf.compute()

        logger.info(f"Completed Dask full read: {len(df)} rows loaded")
        return df

    except Exception as e:
        logger.error(f"Error in Dask CSV reading: {e}")
        raise


def write_dataframe_to_csv(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    encoding: str,
    delimiter: str,
    quotechar: str,
    index: bool,
    show_progress: bool = True,
    compression: Optional[str] = None,
) -> Path:
    """
    Write a DataFrame to a CSV file using Dask.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Path to save the CSV file
    encoding : str
        File encoding
    delimiter : str
        Field delimiter character
    quotechar : str
        Text qualifier character
    index : bool
        Whether to write row indices
    show_progress : bool
        Whether to display a progress bar
    compression : str, optional
        Compression algorithm: 'infer', 'gzip', 'bz2', 'zip', 'xz', 'zstd', or None for no compression

    Returns:
    --------
    Path
        Path to the saved file
    """
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        file_path = Path(file_path)
        logger.info(
            f"Starting Dask write to {file_path} with compression={compression}"
        )

        # Convert to Dask DataFrame with appropriate partitioning
        # Use about 100k rows per partition as a heuristic
        npartitions = max(1, len(df) // 100000)
        ddf = dd.from_pandas(df, npartitions=npartitions)

        logger.debug(f"Created Dask DataFrame with {npartitions} partitions")

        # Write to CSV with progress if requested
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=100,  # Percentage-based progress
                description=f"Writing to {file_path.name} with Dask",
                unit="%",
            )

            # Show initial progress
            progress_bar.update(10)

            # Write with Dask's ProgressBar
            with ProgressBar():
                ddf.to_csv(
                    file_path,
                    encoding=encoding,
                    sep=delimiter,
                    quotechar=quotechar,
                    index=index,
                    single_file=True,
                    compression=compression,  # Pass compression parameter
                )

            # Complete progress bar
            progress_bar.update(90)
            progress_bar.close()
        else:
            # Write without progress display
            ddf.to_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                quotechar=quotechar,
                index=index,
                single_file=True,
                compression=compression,  # Pass compression parameter
            )

        logger.info(f"Completed Dask write to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error in Dask CSV writing: {e}")
        raise


def compute_dask_stats(ddf) -> Dict[str, Any]:
    """
    Compute basic statistics for a Dask DataFrame.

    Parameters:
    -----------
    ddf : dask.dataframe.DataFrame
        Dask DataFrame to analyze

    Returns:
    --------
    Dict[str, Any]
        Dictionary of statistics including:
        - total_rows: Number of rows in the DataFrame
        - partitions: Number of Dask partitions
        - dtypes: Column data types
        - estimated_memory_usage_bytes: Estimated memory usage in bytes
        - estimated_memory_usage_mb: Estimated memory usage in MB
    """
    try:
        logger.debug("Computing Dask DataFrame statistics")
        stats = {}

        # Total number of rows
        stats["total_rows"] = len(ddf)

        # Number of partitions
        stats["partitions"] = ddf.npartitions

        # Column types
        stats["dtypes"] = {str(col): str(dtype) for col, dtype in ddf.dtypes.items()}

        # Column memory usage estimates
        try:
            # Calculate memory usage per partition
            mem_usage = ddf.memory_usage_per_partition().compute()
            total_memory = mem_usage.sum()

            stats["estimated_memory_usage_bytes"] = int(total_memory)
            stats["estimated_memory_usage_mb"] = round(total_memory / (1024 * 1024), 2)

            # Additional memory statistics
            stats["avg_memory_per_partition_mb"] = round(
                total_memory / (1024 * 1024 * ddf.npartitions), 2
            )
        except Exception as mem_error:
            # Memory usage calculation might fail for some datasets
            logger.warning(f"Could not calculate memory usage: {mem_error}")
            stats["estimated_memory_usage_bytes"] = "unknown"
            stats["estimated_memory_usage_mb"] = "unknown"
            stats["avg_memory_per_partition_mb"] = "unknown"

        logger.debug(f"Computed Dask stats: {stats}")
        return stats

    except Exception as e:
        logger.error(f"Error computing Dask stats: {e}")
        raise


def check_dask_availability(engine: str = "auto", logger=None) -> bool:
    """
    Check if Dask is available in the current environment.

    Parameters:
    -----------
    engine : str
        The data processing engine requested: "pandas", "dask", or "auto".
    logger : Logger, optional
        Optional logger to print an error message if Dask is explicitly required but not installed.

    Returns:
    --------
    bool
        True if Dask is available, False otherwise.
    """
    try:
        import dask.dataframe as dd

        return True
    except ImportError:
        if engine == "dask" and logger:
            logger.error(
                "Dask explicitly requested but not available. Install with: pip install dask[complete]"
            )
        return False


def should_use_dask(
    df: pd.DataFrame, engine: str, dask_available: bool, max_rows_in_memory: int
) -> bool:
    """
    Decide whether to use Dask based on configuration, Dask availability, and data size.

    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe to evaluate.
    engine : str
        Desired processing engine: "pandas", "dask", or "auto".
    dask_available : bool
        Whether Dask is available in the environment.
    max_rows_in_memory : int
        Threshold for using Pandas in memory. If row count exceeds this, prefer Dask.

    Returns:
    --------
    bool
        True if Dask should be used, False otherwise.
    """
    if not dask_available:
        return False
    if engine == "dask":
        return True
    elif engine == "pandas":
        return False
    return len(df) > max_rows_in_memory


def convert_to_dask(
    df: pd.DataFrame,
    dask_npartitions: int = None,
    dask_partition_size: str = "100MB",
    logger=None,
):
    """
    Convert a pandas DataFrame into a Dask DataFrame with calculated or fixed partitioning.

    Parameters:
    -----------
    df : pd.DataFrame
        Input pandas DataFrame.
    dask_npartitions : int, optional
        Desired number of Dask partitions. If None, will be calculated based on partition size.
    dask_partition_size : str
        Desired partition size (e.g., "100MB", "1GB").
    logger : Logger, optional
        Optional logger for progress/info output.

    Returns:
    --------
    dd.DataFrame
        Converted Dask DataFrame.
    """
    import dask.dataframe as dd

    memory_usage_mb = get_memory_usage(df)["total_mb"]
    partition_size_mb = parse_partition_size(dask_partition_size)
    npartitions = dask_npartitions or max(1, int(memory_usage_mb / partition_size_mb))

    if logger:
        logger.info(f"Converting to Dask DataFrame with {npartitions} partitions")

    return dd.from_pandas(df, npartitions=npartitions), npartitions


def convert_from_dask(ddf):
    """
    Convert a Dask DataFrame back into a pandas DataFrame.

    Parameters:
    -----------
    ddf : dd.DataFrame
        Dask DataFrame to convert.

    Returns:
    --------
    pd.DataFrame
        Resulting pandas DataFrame.
    """
    return ddf.compute()


def parse_partition_size(partition_size: str) -> float:
    """
    Parse a string-based partition size to megabytes (MB).

    Parameters:
    -----------
    partition_size : str
        Partition size in the form "100MB", "1GB", or numeric string (default MB assumed).

    Returns:
    --------
    float
        Partition size in megabytes.
    """
    partition_size = partition_size.upper()
    if partition_size.endswith("GB"):
        return float(partition_size[:-2]) * 1024
    elif partition_size.endswith("MB"):
        return float(partition_size[:-2])
    return float(partition_size)

def get_computed_df(df):
    """
    Return the computed DataFrame if it's a Dask DataFrame,
    otherwise return the original DataFrame.

    Parameters:
    -----------
    df : Union[pd.DataFrame, dd.DataFrame]
        Input DataFrame (can be Pandas or Dask)

    Returns:
    --------
    pd.DataFrame
        The computed (or original) DataFrame
    """
    return df.compute() if isinstance(df, dd.DataFrame) else df