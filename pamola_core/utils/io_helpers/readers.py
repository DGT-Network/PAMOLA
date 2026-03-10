"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
Module: File Reader Implementations
Description: Format-specific readers shared by io.py and helper utilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

import pamola_core.utils.progress as progress
import pamola_core.utils.io_helpers.csv_utils as csv_utils
import pamola_core.utils.io_helpers.dask_utils as dask_utils
import pamola_core.utils.io_helpers.format_utils as format_utils
from pamola_core.errors.exceptions import (
    DependencyMissingError,
    PamolaFileNotFoundError,
)
from pamola_core.utils.io_helpers.temp_files import temporary_decrypted_file

logger = logging.getLogger(__name__)

# Local defaults (mirrors pamola_core.utils.io)
DEFAULT_ENCODING = "utf-8"
DEFAULT_CSV_DELIMITER = ","
DEFAULT_CSV_QUOTECHAR = '"'
LARGE_FILE_THRESHOLD_MB = 500
DASK_THRESHOLD_MB = 200
PROGRESS_CHUNK_SIZE = 10000


def read_full_csv(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    delimiter: str = DEFAULT_CSV_DELIMITER,
    quotechar: str = DEFAULT_CSV_QUOTECHAR,
    show_progress: bool = True,
    use_dask: bool = False,
    encryption_key: Optional[str] = None,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    use_encryption: bool = False,
    encryption_mode: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reads an entire CSV file into a DataFrame.
    For large files, consider using read_csv_in_chunks instead.

    Parameters
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    show_progress : bool
        Whether to display a progress bar (default: True)
    use_dask : bool
        Whether to use Dask for larger-than-memory datasets (default: False)
    encryption_key : str, optional
        Key for decrypting encrypted files
    columns : List[str], optional
        Specific columns to read (reduces memory usage)
    nrows : int, optional
        Maximum number of rows to read
    skiprows : Union[int, List[int]], optional
        Row indices to skip or number of rows to skip from the start
    use_encryption : bool, optional
        Whether to use encryption for reading the file (default: False)
    encryption_mode : str, optional
        Encryption mode to use (default: "simple")

    Returns
    --------
    pd.DataFrame
        DataFrame containing the entire file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise PamolaFileNotFoundError(str(file_path))

    logger.info(f"Reading full file: {file_path} (encoding: {encoding})")

    # Check file size to warn about large files
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > LARGE_FILE_THRESHOLD_MB:
        logger.warning(
            f"File {file_path} is large ({file_size_mb:.1f}MB). "
            f"Consider using read_csv_in_chunks for better memory efficiency."
        )

    # If Dask is enabled and file is large, use it
    if use_dask and file_size_mb > DASK_THRESHOLD_MB and dask_utils.is_dask_available():
        if encryption_key:
            logger.warning(
                "Encryption with Dask is not supported. Falling back to pandas."
            )
        else:
            try:
                logger.info("Using Dask for reading large CSV file")

                df = dask_utils.read_full_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    show_progress=show_progress,
                )

                # Apply filtering after loading if needed
                if columns is not None:
                    df = df[columns]

                if skiprows is not None:
                    if isinstance(skiprows, int):
                        df = df.iloc[skiprows:]
                    else:
                        # Create boolean mask for rows to keep
                        keep_mask = ~pd.Series(range(len(df))).isin(skiprows)
                        df = df.loc[keep_mask].reset_index(drop=True)

                if nrows is not None:
                    df = df.head(nrows)

                logger.info(f"Completed reading {file_path} with Dask: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Error using Dask: {e}. Falling back to pandas.")

    # Handle potential decryption
    start_time = time.time()

    with temporary_decrypted_file(
        file_path=file_path,
        encryption_key=encryption_key,
        encryption_mode=encryption_mode,
    ) as file_to_read:
        try:
            # Prepare CSV reader options
            reader_options = csv_utils.prepare_csv_reader_options(
                encoding=encoding,
                delimiter=delimiter,
                quotechar=quotechar,
                columns=columns,
                nrows=nrows,
                skiprows=skiprows,
                low_memory=False,
            )

            if show_progress and file_size_mb > 10:
                # For large files, read without line counting to avoid double pass
                logger.info("Reading file with progress tracking...")

                # Use pandas with chunksize for progress updates
                chunk_options = reader_options.copy()
                chunk_options["chunksize"] = PROGRESS_CHUNK_SIZE

                chunks = []
                rows_read = 0

                # Create progress bar based on file size
                progress_bar = progress.ProgressBar(
                    total=int(file_size_mb),
                    description=f"Reading {file_path.name}",
                    unit="MB (approx)",
                )

                try:
                    for chunk in pd.read_csv(file_to_read, **chunk_options):
                        chunks.append(chunk)
                        rows_read += len(chunk)

                        # More accurate progress calculation
                        if rows_read > 1000:
                            bytes_per_row = file_path.stat().st_size / rows_read
                            estimated_total_rows = file_path.stat().st_size / bytes_per_row
                            mb_read = (
                                rows_read / estimated_total_rows
                            ) * file_size_mb
                            progress_bar.update(
                                max(1, int(mb_read) - progress_bar.n)
                            )
                        else:
                            # Early stage: approximate progress
                            progress_bar.update(1)

                    # Concatenate chunks
                    df = pd.concat(chunks, ignore_index=True)
                    progress_bar.update(int(file_size_mb) - progress_bar.n)
                    progress_bar.close()
                except Exception:
                    progress_bar.close()
                    raise
            else:
                # Read normally without progress tracking
                df = pd.read_csv(file_to_read, **reader_options)

            duration = time.time() - start_time
            logger.info(
                f"Completed reading {file_path}: {len(df)} rows in {duration:.2f}s"
            )
            return df

        except pd.errors.ParserError as e:
            logger.error(f"CSV parsing error: {e}")
            raise
        except (IOError, OSError) as e:
            logger.error(f"File I/O error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error reading CSV: {e}")
            raise


def read_text(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    sep: str = "\t",
    show_progress: bool = True,
    encryption_key: Optional[str] = None,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a text file (like TSV) into a DataFrame.

    Parameters
    -----------
    file_path : str or Path
        Path to the text file
    encoding : str
        File encoding (default: "utf-8")
    sep : str
        Separator/delimiter character (default: '\t')
    show_progress : bool
        Whether to display a progress bar (default: True)
    encryption_key : str, optional
        Key for decrypting encrypted files
    columns : List[str], optional
        Specific columns to include (reduces memory usage)
    nrows : int, optional
        Maximum number of rows to read
    skiprows : Union[int, List[int]], optional
        Row indices to skip or number of rows to skip from the start
    **kwargs
        Additional arguments passed to pandas.read_csv

    Returns
    --------
    pd.DataFrame
        DataFrame containing the file data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise PamolaFileNotFoundError(str(file_path))

    logger.info(f"Reading text file: {file_path} with separator '{sep}'")
    start_time = time.time()

    # Handle potential decryption
    with temporary_decrypted_file(file_path, encryption_key) as file_to_read:
        try:
            # Prepare CSV reader options with tab as separator
            reader_options = csv_utils.prepare_csv_reader_options(
                encoding=encoding,
                delimiter=sep,
                columns=columns,
                nrows=nrows,
                skiprows=skiprows,
                low_memory=False,
            )

            # Add any additional kwargs
            reader_options.update(kwargs)

            if show_progress:
                # Get file size for progress estimation
                file_size_mb = file_to_read.stat().st_size / (1024 * 1024)

                if file_size_mb > 10:
                    logger.info("Reading file with progress tracking...")

                    # Read in chunks to update progress
                    chunks = []
                    chunk_options = reader_options.copy()
                    chunk_options["chunksize"] = PROGRESS_CHUNK_SIZE

                    # Create progress bar based on file size
                    progress_bar = progress.ProgressBar(
                        total=int(file_size_mb),
                        description=f"Reading {file_path.name}",
                        unit="MB (approx)",
                    )

                    try:
                        rows_read = 0
                        for chunk in pd.read_csv(file_to_read, **chunk_options):
                            chunks.append(chunk)
                            rows_read += len(chunk)

                            # More accurate progress calculation
                            if rows_read > 1000:
                                bytes_per_row = file_path.stat().st_size / rows_read
                                estimated_total_rows = (
                                    file_path.stat().st_size / bytes_per_row
                                )
                                mb_read = (
                                    rows_read / estimated_total_rows
                                ) * file_size_mb
                                progress_bar.update(
                                    max(1, int(mb_read) - progress_bar.n)
                                )
                            else:
                                progress_bar.update(1)

                        # Concatenate chunks
                        df = pd.concat(chunks, ignore_index=True)
                        progress_bar.update(int(file_size_mb) - progress_bar.n)
                        progress_bar.close()
                    except Exception:
                        progress_bar.close()
                        raise
                else:
                    # Small file: read normally
                    df = pd.read_csv(file_to_read, **reader_options)
            else:
                # Read normally without progress tracking
                df = pd.read_csv(file_to_read, **reader_options)

            duration = time.time() - start_time
            logger.info(
                f"Completed reading text file {file_path}: {len(df)} rows in {duration:.2f}s"
            )
            return df

        except pd.errors.ParserError as e:
            logger.error(f"Text file parsing error: {e}")
            raise
        except (IOError, OSError) as e:
            logger.error(f"File I/O error: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error reading text file: {e}")
            raise


def read_excel(
    file_path: Union[str, Path],
    sheet_name: Union[str, int, None] = 0,
    show_progress: bool = True,
    encryption_key: Optional[str] = None,
    columns: Optional[List[str]] = None,
    nrows: Optional[int] = None,
    skiprows: Optional[Union[int, List[int]]] = None,
    use_encryption: bool = False,
    encryption_mode: Optional[str] = None,
    **kwargs,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Reads an Excel file into a DataFrame.

    Parameters
    -----------
    file_path : str or Path
        Path to the Excel file
    sheet_name : str, int, or None
        Name or index of the sheet to read (default: 0, the first sheet)
        If None, all sheets are read into a dictionary of DataFrames
    show_progress : bool
        Whether to display a progress bar (default: True)
    encryption_key : str, optional
        Key for decrypting encrypted files
    columns : List[str], optional
        Specific columns to include (reduces memory usage)
    nrows : int, optional
        Maximum number of rows to read
    skiprows : Union[int, List[int]], optional
        Row indices to skip or number of rows to skip from the start
    use_encryption : bool, optional
        Whether to use encryption for reading the file (default: False)
    encryption_mode : str, optional
        Encryption mode to use (default: "simple")
    **kwargs
        Additional arguments passed to pandas.read_excel

    Returns
    --------
    Union[pd.DataFrame, Dict[str, pd.DataFrame]]
        DataFrame containing the file data, or dictionary of DataFrames if sheet_name=None
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise PamolaFileNotFoundError(str(file_path))

    # Check if openpyxl is available
    format_utils.check_openpyxl_available()

    logger.info(f"Reading Excel file: {file_path}")
    start_time = time.time()

    # Handle potential decryption
    with temporary_decrypted_file(
        file_path, encryption_key, suffix=".xlsx", encryption_mode=encryption_mode
    ) as file_to_read:
        try:
            # Prepare Excel reader options
            excel_options = {}

            # Add nrows and skiprows if specified
            if nrows is not None:
                excel_options["nrows"] = nrows

            if skiprows is not None:
                excel_options["skiprows"] = skiprows

            # Add any additional kwargs
            excel_options.update(kwargs)

            # If sheet_name is None, pandas returns a dict of all sheets
            # We need to handle that case differently for progress display
            if sheet_name is None:
                if show_progress:
                    # First, get the sheet names to know how many sheets to process
                    sheet_names = pd.ExcelFile(file_to_read).sheet_names

                    # Create progress bar for sheets
                    progress_bar = progress.ProgressBar(
                        total=len(sheet_names),
                        description=f"Reading sheets from {file_path.name}",
                        unit="sheets",
                    )

                    # Read each sheet individually with progress updates
                    result = {}
                    for sheet in sheet_names:
                        df = pd.read_excel(
                            file_to_read, sheet_name=sheet, **excel_options
                        )

                        # Filter columns if requested
                        if columns is not None:
                            valid_cols = [col for col in columns if col in df.columns]
                            if valid_cols:
                                df = df[valid_cols]

                        result[sheet] = df
                        progress_bar.update(1, postfix={"sheet": sheet})

                    progress_bar.close()

                    # Calculate total rows read
                    total_rows = sum(len(df) for df in result.values())
                    logger.info(
                        f"Read {len(result)} sheets with {total_rows} total rows"
                    )

                else:
                    # Without progress tracking, read all sheets at once
                    result = pd.read_excel(
                        file_to_read, sheet_name=None, **excel_options
                    )

                    # Filter columns if requested
                    if columns is not None:
                        for sheet_name, df in result.items():
                            valid_cols = [col for col in columns if col in df.columns]
                            if valid_cols:
                                result[sheet_name] = df[valid_cols]
            else:
                # Reading a single sheet
                result = pd.read_excel(
                    file_to_read, sheet_name=sheet_name, **excel_options
                )

                # Filter columns if requested
                if columns is not None and isinstance(result, pd.DataFrame):
                    valid_cols = [col for col in columns if col in result.columns]
                    if valid_cols:
                        result = result[valid_cols]

                if show_progress and isinstance(result, pd.DataFrame):
                    logger.info(f"Read sheet with {len(result)} rows")

            duration = time.time() - start_time
            logger.info(f"Completed reading Excel file {file_path} in {duration:.2f}s")
            return result

        except ImportError as e:
            logger.error(f"Missing Excel library dependency: {e}")
            raise DependencyMissingError(
                dependency_name="openpyxl",
                reason=str(e),
            ) from e
        except Exception as e:
            logger.exception(f"Unexpected error reading Excel file {file_path}: {e}")
            raise


def read_json(
    file_path: Union[str, Path],
    encoding: str = DEFAULT_ENCODING,
    encryption_key: Optional[str] = None,
    use_encryption: bool = False,
    encryption_mode: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Reads a JSON file into a dictionary.

    Parameters
    -----------
    file_path : str or Path
        Path to the JSON file
    encoding : str
        File encoding (default: "utf-8")
    encryption_key : str, optional
        Key for decrypting the file
    use_encryption : bool, optional
        Whether to use encryption for reading the file (default: False)
    encryption_mode : str, optional
        Encryption mode to use (default: "simple")
    **kwargs
        Additional arguments passed to json.loads

    Returns
    --------
    Dict[str, Any]
        Dictionary containing the JSON data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise PamolaFileNotFoundError(str(file_path))

    logger.info(f"Reading JSON file: {file_path}")
    start_time = time.time()

    try:
        # Handle decryption if needed
        with temporary_decrypted_file(
            file_path, encryption_key, suffix=".json", encryption_mode=encryption_mode
        ) as file_to_read:
            # Read the file
            with open(file_to_read, "r", encoding=encoding) as f:
                content = f.read()

            # Parse JSON
            data = json.loads(content, **kwargs)

            duration = time.time() - start_time
            logger.info(f"Read JSON file {file_path} in {duration:.2f}s")
            return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {file_path}: {e}")
        raise
    except (IOError, OSError) as e:
        logger.error(f"File I/O error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error reading JSON file {file_path}: {e}")
        raise


def read_parquet(
    file_path: Union[str, Path],
    columns: Optional[List[str]] = None,
    encryption_key: Optional[str] = None,
    use_encryption: bool = False,
    encryption_mode: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Reads a Parquet file into a DataFrame.

    Parameters
    -----------
    file_path : str or Path
        Path to the Parquet file
    columns : List[str], optional
        Specific columns to read (reduces memory usage)
    encryption_key : str, optional
        Key for decrypting the file
    use_encryption : bool, optional
        Whether to use encryption for reading the file (default: False)
    encryption_mode : str, optional
        Encryption mode to use (default: "simple")
    **kwargs
        Additional arguments to pass to pandas.read_parquet

    Returns
    --------
    pd.DataFrame
        DataFrame containing the file data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise PamolaFileNotFoundError(str(file_path))

    logger.info(f"Reading Parquet file: {file_path}")
    start_time = time.time()

    try:
        # Check if pyarrow is installed
        format_utils.check_pyarrow_available()

        # Handle decryption if needed
        with temporary_decrypted_file(
            file_path,
            encryption_key,
            suffix=".parquet",
            encryption_mode=encryption_mode,
        ) as file_to_read:
            # Read the file
            df = pd.read_parquet(file_to_read, columns=columns, **kwargs)

            duration = time.time() - start_time
            logger.info(f"Read Parquet file {file_path} in {duration:.2f}s")
            return df

    except ImportError as e:
        logger.error(f"Missing Parquet library dependency: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error reading Parquet file {file_path}: {e}")
        raise
