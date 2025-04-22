"""
Utilities for integration with Dask distributed computing framework.

This module provides functions for using Dask with the HHR I/O system,
enabling processing of larger-than-memory datasets. It isolates Dask-specific
code to avoid dependencies in the main I/O module when Dask is not being used.
"""

from pathlib import Path
from typing import Iterator, Union, Dict, Any

import pandas as pd

from pamola_core.utils import logging
from pamola_core.utils import progress

# Configure module logger
logger = logging.get_logger("hhr.utils.io_helpers.dask_utils")


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
        return True
    except ImportError:
        return False


def read_csv_in_chunks(file_path: Union[str, Path],
                       chunk_size: int,
                       encoding: str,
                       delimiter: str,
                       quotechar: str,
                       show_progress: bool = True) -> Iterator[pd.DataFrame]:
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

        # Read with Dask
        ddf = dd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            quotechar=quotechar,
            blocksize=chunk_size * 100  # Adjust blocksize for Dask
        )

        # Get total number of partitions
        n_partitions = ddf.npartitions

        # Show progress if requested
        progress_bar = None
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=n_partitions,
                description=f"Reading {file_path.name} with Dask",
                unit="partitions"
            )

        # Process in chunks
        for i in range(0, len(ddf), chunk_size):
            end = min(i + chunk_size, len(ddf))

            # Update progress
            if progress_bar:
                progress_bar.update(1)

            # Compute chunk
            chunk = ddf.iloc[i:end].compute()

            logger.debug(f"Read Dask chunk {i // chunk_size + 1}: {len(chunk)} rows")
            yield chunk

        # Close progress bar
        if progress_bar:
            progress_bar.close()

    except Exception as e:
        logger.error(f"Error in Dask CSV reading: {e}")
        raise


def read_full_csv(file_path: Union[str, Path],
                  encoding: str,
                  delimiter: str,
                  quotechar: str,
                  show_progress: bool = True) -> pd.DataFrame:
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

        # Read with Dask
        ddf = dd.read_csv(
            file_path,
            encoding=encoding,
            sep=delimiter,
            quotechar=quotechar
        )

        # Compute to pandas DataFrame with progress if requested
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=100,  # Percentage-based progress
                description=f"Reading {file_path.name} with Dask",
                unit="%"
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
            df = ddf.compute()

        return df

    except Exception as e:
        logger.error(f"Error in Dask CSV reading: {e}")
        raise


def write_dataframe_to_csv(df: pd.DataFrame,
                           file_path: Union[str, Path],
                           encoding: str,
                           delimiter: str,
                           quotechar: str,
                           index: bool,
                           show_progress: bool = True) -> Path:
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

    Returns:
    --------
    Path
        Path to the saved file
    """
    try:
        import dask.dataframe as dd
        from dask.diagnostics import ProgressBar

        file_path = Path(file_path)

        # Convert to Dask DataFrame with appropriate partitioning
        # Use about 100k rows per partition as a heuristic
        npartitions = max(1, len(df) // 100000)
        ddf = dd.from_pandas(df, npartitions=npartitions)

        # Write to CSV with progress if requested
        if show_progress:
            progress_bar = progress.ProgressBar(
                total=100,  # Percentage-based progress
                description=f"Writing to {file_path.name} with Dask",
                unit="%"
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
                    single_file=True
                )

            # Complete progress bar
            progress_bar.update(90)
            progress_bar.close()
        else:
            ddf.to_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                quotechar=quotechar,
                index=index,
                single_file=True
            )

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
        Dictionary of statistics
    """
    try:
        stats = {}

        # Total number of rows
        stats['total_rows'] = len(ddf)

        # Number of partitions
        stats['partitions'] = ddf.npartitions

        # Column types
        stats['dtypes'] = {str(col): str(dtype) for col, dtype in ddf.dtypes.items()}

        # Column memory usage estimates
        try:
            mem_usage = ddf.memory_usage_per_partition().compute()
            stats['estimated_memory_usage_bytes'] = int(mem_usage.sum())
            stats['estimated_memory_usage_mb'] = round(mem_usage.sum() / (1024 * 1024), 2)
        except:
            # Memory usage calculation might fail for some datasets
            stats['estimated_memory_usage'] = 'unknown'

        return stats

    except Exception as e:
        logger.error(f"Error computing Dask stats: {e}")
        raise