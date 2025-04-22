"""
Core I/O Utilities
----------------------------------------------
This module provides robust file handling capabilities optimized for
processing datasets of various sizes, from small to very large.

Features:
- High-performance CSV reading with chunking and progress tracking
- Support for large datasets via Dask integration
- Customizable encoding, delimiters, and text qualifiers
- Memory-efficient processing
- JSON handling with customizable formatting
- Integration with encryption/decryption capabilities
- Support for CSV, JSON, Parquet, Excel, Pickle, and image formats
- Directory creation as needed

All operations receive explicit paths and parameters from calling code,
without relying on internal configuration or implicit paths.

(C) 2025 BDA

Author: V.Khvatov (original), Updated in refactoring
"""

import time
from pathlib import Path
from typing import Dict, List, Union, Optional, Iterator, Any, TypeVar
import plotly.graph_objects as go
from PIL import Image

import pandas as pd

from pamola_core.utils import logging
from pamola_core.utils import progress

# Import helpers
from pamola_core.utils.io_helpers import dask_utils
from pamola_core.utils.io_helpers import crypto_utils
from pamola_core.utils.io_helpers import format_utils
from pamola_core.utils.io_helpers import directory_utils
from pamola_core.utils.io_helpers import csv_utils
from pamola_core.utils.io_helpers import json_utils

# Configure module logger
logger = logging.get_logger("hhr.utils.io")

# Type variables for generic functions
T = TypeVar('T')
DataFrame = TypeVar('DataFrame', bound=pd.DataFrame)


# ====================
# Directory Management Functions
# ====================

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    Ensures the specified directory exists, creating it if necessary.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory to ensure exists

    Returns:
    --------
    Path
        Pathlib object for the ensured directory
    """
    return directory_utils.ensure_directory(directory)


def get_timestamped_filename(base_name: str, extension: str = "csv",
                             include_timestamp: bool = True) -> str:
    """
    Creates a timestamped filename.

    Parameters:
    -----------
    base_name : str
        Base name for the file
    extension : str
        File extension (default: "csv")
    include_timestamp : bool
        Whether to include a timestamp in the filename (default: True)

    Returns:
    --------
    str
        Timestamped filename
    """
    return directory_utils.get_timestamped_filename(base_name, extension, include_timestamp)


def get_file_stats(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Gets statistics about a file.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    Dict[str, Any]
        Dictionary with file statistics (size, creation time, modification time, etc.)
    """
    return directory_utils.get_file_stats(file_path)


def list_directory_contents(directory: Union[str, Path],
                            pattern: str = "*",
                            recursive: bool = False) -> List[Path]:
    """
    Lists the contents of a directory.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory
    pattern : str
        Glob pattern for filtering files (default: "*")
    recursive : bool
        Whether to search recursively (default: False)

    Returns:
    --------
    List[Path]
        List of paths to the files matching the pattern
    """
    return directory_utils.list_directory_contents(directory, pattern, recursive)


def clear_directory(directory: Union[str, Path],
                    ignore_patterns: Optional[List[str]] = None,
                    confirm: bool = True) -> int:
    """
    Clears all files and subdirectories in the specified directory.

    Parameters:
    -----------
    directory : str or Path
        Path to the directory to clear
    ignore_patterns : List[str], optional
        List of glob patterns to ignore
    confirm : bool
        Whether to ask for confirmation before clearing (default: True)

    Returns:
    --------
    int
        Number of items removed
    """
    return directory_utils.clear_directory(directory, ignore_patterns, confirm)


# ====================
# CSV Reading Functions
# ====================

def read_csv_in_chunks(file_path: Union[str, Path],
                       chunk_size: int = 100000,
                       encoding: str = "utf-16",
                       delimiter: str = ",",
                       quotechar: str = '"',
                       show_progress: bool = True,
                       use_dask: bool = False,
                       encryption_key: Optional[str] = None) -> Iterator[pd.DataFrame]:
    """
    Reads a very large CSV file in chunks, yielding each chunk as a DataFrame.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    chunk_size : int
        Number of rows to read per chunk (default: 100,000)
    encoding : str
        File encoding (default: "utf-16")
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

    Yields:
    -------
    pd.DataFrame
        DataFrame containing each chunk of data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Starting to read file: {file_path} (chunk size: {chunk_size}, encoding: {encoding})")
    start_time = time.time()
    total_rows = 0

    # If Dask is requested and available, use it for large files
    if use_dask and dask_utils.is_dask_available():
        # Dask doesn't support encryption yet
        if encryption_key:
            logger.warning("Encryption with Dask is not supported. Falling back to pandas.")
        else:
            try:
                for chunk in dask_utils.read_csv_in_chunks(
                        file_path,
                        chunk_size=chunk_size,
                        encoding=encoding,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        show_progress=show_progress
                ):
                    total_rows += len(chunk)
                    yield chunk

                duration = time.time() - start_time
                logger.info(f"Completed reading {file_path} with Dask: {total_rows} total rows in {duration:.2f}s")
                return
            except Exception as e:
                logger.warning(f"Error using Dask: {e}. Falling back to pandas chunking.")

    # Handle potential decryption
    temp_file_var = None
    if encryption_key:
        logger.info("Decryption requested for file reading")
        # For encrypted files, we'll first decrypt to a temporary file
        try:
            temp_file_var = crypto_utils.decrypt_file(file_path, encryption_key)
            file_to_read = temp_file_var
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    else:
        file_to_read = file_path

    # Prepare progress tracking
    total_lines = None
    if show_progress:
        try:
            total_lines = csv_utils.count_csv_lines(file_to_read, encoding)
            if total_lines > 0:  # Account for header
                total_lines -= 1
            total_chunks = (total_lines // chunk_size) + 1
        except Exception as e:
            logger.warning(f"Could not count lines in file: {e}. Progress bar will not show total.")
            total_chunks = None
    else:
        total_chunks = None

    # Create progress bar
    progress_bar = None
    if show_progress:
        progress_bar = progress.ProgressBar(
            total=total_chunks,
            description=f"Reading {file_path.name}",
            unit="chunks"
        )

    # Prepare CSV reader options
    reader_options = csv_utils.prepare_csv_reader_options(
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
        chunksize=chunk_size,
        low_memory=False
    )

    # Read in chunks
    try:
        chunks_iterator = pd.read_csv(file_to_read, **reader_options)

        for chunk_idx, chunk in enumerate(chunks_iterator):
            chunk_rows = len(chunk)
            total_rows += chunk_rows

            if progress_bar:
                memory_usage = csv_utils.report_memory_usage()
                progress_bar.update(1, postfix={
                    "rows": total_rows,
                    "mem": f"{memory_usage['rss_mb']:.1f}MB"
                })

            logger.debug(f"Read chunk {chunk_idx + 1}: {chunk_rows} rows")
            yield chunk

    finally:
        # Clean up
        if progress_bar:
            progress_bar.close()

        # Remove temporary decrypted file if it was created
        if encryption_key:
            crypto_utils.safe_remove_temp_file(temp_file_var, logger)

        duration = time.time() - start_time
        logger.info(f"Completed reading {file_path}: {total_rows} total rows in {duration:.2f}s")


def read_full_csv(file_path: Union[str, Path],
                  encoding: str = "utf-16",
                  delimiter: str = ",",
                  quotechar: str = '"',
                  show_progress: bool = True,
                  use_dask: bool = False,
                  encryption_key: Optional[str] = None) -> pd.DataFrame:
    """
    Reads an entire CSV file into a DataFrame.
    For large files, consider using read_csv_in_chunks instead.

    Parameters:
    -----------
    file_path : str or Path
        Path to the CSV file
    encoding : str
        File encoding (default: "utf-16")
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

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the entire file
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading full file: {file_path} (encoding: {encoding})")

    # Check file size to warn about large files
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 500:  # If file is larger than 500MB
        logger.warning(
            f"File {file_path} is large ({file_size_mb:.1f}MB). "
            f"Consider using read_csv_in_chunks for better memory efficiency."
        )

    # If Dask is enabled and file is large, use it
    if use_dask and file_size_mb > 200 and dask_utils.is_dask_available():
        if encryption_key:
            logger.warning("Encryption with Dask is not supported. Falling back to pandas.")
        else:
            try:
                logger.info("Using Dask for reading large CSV file")
                df = dask_utils.read_full_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    show_progress=show_progress
                )
                logger.info(f"Completed reading {file_path} with Dask: {len(df)} rows")
                return df
            except Exception as e:
                logger.warning(f"Error using Dask: {e}. Falling back to pandas.")

    # Handle potential decryption
    temp_file_var = None
    if encryption_key:
        logger.info("Decryption requested for file reading")
        try:
            temp_file_var = crypto_utils.decrypt_file(file_path, encryption_key)
            file_to_read = temp_file_var
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    else:
        file_to_read = file_path

    start_time = time.time()

    # Prepare CSV reader options
    reader_options = csv_utils.prepare_csv_reader_options(
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
        low_memory=False
    )

    if show_progress:
        # For progress display, we'll read line count first
        try:
            total_lines = csv_utils.count_csv_lines(file_to_read, encoding)

            # Create progress bar
            progress_bar = progress.ProgressBar(
                total=total_lines,
                description=f"Reading {file_path.name}",
                unit="lines"
            )

            # Read in chunks to update progress
            chunks = []
            chunk_options = reader_options.copy()
            chunk_options['chunksize'] = 10000

            for chunk in pd.read_csv(file_to_read, **chunk_options):
                chunks.append(chunk)
                progress_bar.update(len(chunk))

            # Combine chunks
            df = pd.concat(chunks, ignore_index=True)
            progress_bar.close()
        except Exception as e:
            logger.warning(f"Progress bar creation failed: {e}. Reading file without progress tracking.")
            # Fallback to normal reading
            df = pd.read_csv(file_to_read, **reader_options)
    else:
        # Read normally without progress tracking
        df = pd.read_csv(file_to_read, **reader_options)

    # Clean up temporary file if created
    if encryption_key:
        crypto_utils.safe_remove_temp_file(temp_file_var, logger)

    duration = time.time() - start_time
    logger.info(f"Completed reading {file_path}: {len(df)} rows in {duration:.2f}s")

    return df


# ====================
# CSV Writing Functions
# ====================

def write_dataframe_to_csv(df: pd.DataFrame,
                           file_path: Union[str, Path],
                           encoding: str = "utf-16",
                           delimiter: str = ",",
                           quotechar: str = '"',
                           index: bool = False,
                           show_progress: bool = True,
                           use_dask: bool = False,
                           encryption_key: Optional[str] = None) -> Path:
    """
    Writes a DataFrame to a CSV file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Path to save the CSV file
    encoding : str
        File encoding (default: "utf-16")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    index : bool
        Whether to write row indices (default: False)
    show_progress : bool
        Whether to display a progress bar (default: True)
    use_dask : bool
        Whether to use Dask for larger datasets (default: False)
    encryption_key : str, optional
        Key for encrypting the file

    Returns:
    --------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    logger.info(f"Writing DataFrame to {file_path} ({len(df)} rows)")
    start_time = time.time()

    # If encryption is requested, we'll write to a temporary file first
    temp_file = None
    if encryption_key:
        from tempfile import NamedTemporaryFile
        temp_file = NamedTemporaryFile(delete=False, suffix='.csv')
        output_path = Path(temp_file.name)
        logger.info(f"Encryption requested. Writing to temporary file first: {output_path}")
    else:
        output_path = file_path

    try:
        # If Dask is enabled and DataFrame is large, use it
        if use_dask and len(df) > 500000 and dask_utils.is_dask_available() and not encryption_key:
            try:
                logger.info("Using Dask for writing large DataFrame")
                dask_utils.write_dataframe_to_csv(
                    df,
                    output_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    index=index,
                    show_progress=show_progress
                )
                logger.info(f"Wrote DataFrame to {output_path} with Dask")
            except Exception as e:
                logger.warning(f"Error using Dask: {e}. Falling back to pandas.")
                # Continue to regular pandas approach below

        # For large DataFrames with progress bar
        elif show_progress and len(df) > 10000:
            try:
                # Create progress bar
                progress_bar = progress.ProgressBar(
                    total=len(df),
                    description=f"Writing to {file_path.name}",
                    unit="rows"
                )

                # Prepare CSV writer options
                writer_options = csv_utils.prepare_csv_writer_options(
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    index=index
                )

                # Split into chunks for progress updates
                chunk_size = 50000
                num_chunks = (len(df) + chunk_size - 1) // chunk_size

                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size]
                    mode = 'w' if i == 0 else 'a'
                    header = i == 0

                    chunk_options = writer_options.copy()
                    chunk_options['mode'] = mode
                    chunk_options['header'] = header

                    chunk.to_csv(output_path, **chunk_options)
                    progress_bar.update(len(chunk))

                progress_bar.close()
            except Exception as e:
                logger.warning(f"Error with chunked writing: {e}. Falling back to standard method.")

                # Prepare CSV writer options
                writer_options = csv_utils.prepare_csv_writer_options(
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    index=index
                )

                df.to_csv(output_path, **writer_options)
        else:
            # Standard write for smaller DataFrames or when progress is not needed
            writer_options = csv_utils.prepare_csv_writer_options(
                encoding=encoding,
                delimiter=delimiter,
                quotechar=quotechar,
                index=index
            )

            df.to_csv(output_path, **writer_options)

        # If encryption was requested, encrypt the temporary file and save to the target path
        if encryption_key and temp_file:
            logger.info(f"Encrypting and saving to final destination: {file_path}")
            crypto_utils.encrypt_file(output_path, file_path, encryption_key)

            # Close and remove temporary file
            try:
                if temp_file:
                    temp_file.close()
                import os
                os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {e}")

        duration = time.time() - start_time
        logger.info(f"Wrote {len(df)} rows to {file_path} in {duration:.2f}s")

        return file_path

    except Exception as e:
        # Clean up on error
        if temp_file:
            try:
                temp_file.close()
                import os
                os.unlink(output_path)
            except:
                pass
        logger.error(f"Error writing DataFrame to CSV: {e}")
        raise


def write_chunks_to_csv(chunks: Iterator[pd.DataFrame],
                        file_path: Union[str, Path],
                        encoding: str = "utf-16",
                        delimiter: str = ",",
                        quotechar: str = '"',
                        index: bool = False,
                        encryption_key: Optional[str] = None) -> Path:
    """
    Writes an iterator of DataFrame chunks to a CSV file.

    Parameters:
    -----------
    chunks : Iterator[pd.DataFrame]
        Iterator of DataFrame chunks to write
    file_path : str or Path
        Path to save the CSV file
    encoding : str
        File encoding (default: "utf-16")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    index : bool
        Whether to write row indices (default: False)
    encryption_key : str, optional
        Key for encrypting the file

    Returns:
    --------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    logger.info(f"Writing chunks to {file_path}")
    start_time = time.time()
    total_rows = 0

    # If encryption is requested, we'll write to a temporary file first
    temp_file = None
    if encryption_key:
        from tempfile import NamedTemporaryFile
        temp_file = NamedTemporaryFile(delete=False, suffix='.csv')
        output_path = Path(temp_file.name)
        logger.info(f"Encryption requested. Writing to temporary file first: {output_path}")
    else:
        output_path = file_path

    try:
        # Initialize the progress bar
        progress_bar = progress.ProgressBar(
            description=f"Writing chunks to {file_path.name}",
            unit="chunks"
        )

        # Write chunks one by one
        for i, chunk in enumerate(chunks):
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False

            # Prepare CSV writer options
            writer_options = csv_utils.prepare_csv_writer_options(
                encoding=encoding,
                delimiter=delimiter,
                quotechar=quotechar,
                index=index,
                mode=mode,
                header=header
            )

            # Write this chunk
            chunk.to_csv(output_path, **writer_options)

            chunk_rows = len(chunk)
            total_rows += chunk_rows

            # Update progress
            progress_bar.update(1, postfix={"total_rows": total_rows})

        progress_bar.close()

        # If encryption was requested, encrypt the temporary file and save to the target path
        if encryption_key and temp_file:
            logger.info(f"Encrypting and saving to final destination: {file_path}")
            crypto_utils.encrypt_file(output_path, file_path, encryption_key)

            # Close and remove temporary file
            try:
                if temp_file:
                    temp_file.close()
                import os
                os.unlink(output_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {e}")

        duration = time.time() - start_time
        logger.info(f"Wrote {total_rows} total rows to {file_path} in {duration:.2f}s")

        return file_path

    except Exception as e:
        # Clean up on error
        if temp_file:
            try:
                temp_file.close()
                import os
                os.unlink(output_path)
            except:
                pass
        logger.error(f"Error writing chunks to CSV: {e}")
        raise


# ====================
# JSON Reading and Writing Functions
# ====================

def read_json(file_path: Union[str, Path],
              encoding: str = "utf-8",
              encryption_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Reads a JSON file into a dictionary.

    Parameters:
    -----------
    file_path : str or Path
        Path to the JSON file
    encoding : str
        File encoding (default: "utf-8")
    encryption_key : str, optional
        Key for decrypting the file

    Returns:
    --------
    Dict[str, Any]
        Dictionary containing the JSON data
    """
    import json

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading JSON file: {file_path}")
    start_time = time.time()

    try:
        # Handle decryption if needed
        if encryption_key:
            try:
                content = crypto_utils.decrypt_file(file_path, encryption_key)
            except Exception as e:
                logger.error(f"Decryption failed: {e}")
                raise
        else:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()

        # Parse JSON
        data = json.loads(content)

        duration = time.time() - start_time
        logger.info(f"Read JSON file {file_path} in {duration:.2f}s")
        return data

    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error in {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        raise


def write_json(data: Union[Dict[str, Any], List[Any]],
              file_path: Union[str, Path],
              encoding: str = "utf-8",
              indent: int = 2,
              ensure_ascii: bool = False,
              convert_numpy: bool = True,
              encryption_key: Optional[str] = None) -> Path:
    """
    Writes a dictionary to a JSON file.

    Parameters:
    -----------
    data : Dict[str, Any]
        Dictionary to write
    file_path : str or Path
        Path to save the JSON file
    encoding : str
        File encoding (default: "utf-8")
    indent : int
        Number of spaces for indentation (default: 2)
    ensure_ascii : bool
        Whether to escape non-ASCII characters (default: False)
    convert_numpy : bool
        Whether to convert NumPy types to standard Python types (default: True)
    encryption_key : str, optional
        Key for encrypting the file

    Returns:
    --------
    Path
        Path to the saved file
    """
    import json

    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    logger.info(f"Writing JSON to {file_path}")
    start_time = time.time()

    try:
        # Convert NumPy types if needed
        if convert_numpy:
            data = json_utils.convert_numpy_types(data)

        # Prepare JSON writer options
        writer_options = json_utils.prepare_json_writer_options(
            ensure_ascii=ensure_ascii,
            indent=indent
        )

        # Convert to JSON string
        json_content = json.dumps(data, **writer_options)

        # Handle encryption if needed
        if encryption_key:
            try:
                crypto_utils.encrypt_content_to_file(json_content, file_path, encryption_key)
            except Exception as e:
                logger.error(f"Encryption failed: {e}")
                raise
        else:
            # Write unencrypted JSON
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(json_content)

        duration = time.time() - start_time
        logger.info(f"Wrote JSON to {file_path} in {duration:.2f}s")
        return file_path

    except Exception as e:
        logger.error(f"Error writing JSON to {file_path}: {e}")
        raise


def append_to_json_array(item: Dict[str, Any],
                         file_path: Union[str, Path],
                         encoding: str = "utf-8",
                         indent: int = 2,
                         convert_numpy: bool = True,
                         create_if_missing: bool = True,
                         encryption_key: Optional[str] = None) -> Path:
    """
    Appends an item to a JSON array file. If the file doesn't exist or doesn't
    contain a valid JSON array, a new array is created.

    Parameters:
    -----------
    item : Dict[str, Any]
        Item to append to the array
    file_path : str or Path
        Path to the JSON file
    encoding : str
        File encoding (default: "utf-8")
    indent : int
        Number of spaces for indentation (default: 2)
    convert_numpy : bool
        Whether to convert NumPy types to standard Python types (default: True)
    create_if_missing : bool
        Whether to create the file if it doesn't exist (default: True)
    encryption_key : str, optional
        Key for encrypting/decrypting the file

    Returns:
    --------
    Path
        Path to the saved file
    """
    import json

    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    # Convert NumPy types if needed
    if convert_numpy:
        item = json_utils.convert_numpy_types(item)

    # Read existing data or create new array
    current_data = []
    if file_path.exists():
        try:
            # Try to read the existing file
            current_data = read_json(file_path, encoding, encryption_key)

            # Ensure it's an array
            if not isinstance(current_data, list):
                logger.warning(f"Existing file {file_path} does not contain a JSON array. Creating new array.")
                current_data = []
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading existing JSON from {file_path}: {e}. Creating new array.")
            current_data = []
    elif not create_if_missing:
        raise FileNotFoundError(f"File not found: {file_path} and create_if_missing is False")

    # Append the new item
    current_data.append(item)

    # Write back to the file
    return write_json(
        current_data,
        file_path,
        encoding=encoding,
        indent=indent,
        convert_numpy=False,  # Already converted
        encryption_key=encryption_key
    )


def merge_json_objects(item: Dict[str, Any],
                       file_path: Union[str, Path],
                       encoding: str = "utf-8",
                       indent: int = 2,
                       convert_numpy: bool = True,
                       create_if_missing: bool = True,
                       overwrite_existing: bool = True,
                       recursive_merge: bool = False,
                       encryption_key: Optional[str] = None) -> Path:
    """
    Merges a dictionary with an existing JSON object file. If the file doesn't exist,
    a new JSON object is created.

    Parameters:
    -----------
    item : Dict[str, Any]
        Dictionary to merge with existing JSON object
    file_path : str or Path
        Path to the JSON file
    encoding : str
        File encoding (default: "utf-8")
    indent : int
        Number of spaces for indentation (default: 2)
    convert_numpy : bool
        Whether to convert NumPy types to standard Python types (default: True)
    create_if_missing : bool
        Whether to create the file if it doesn't exist (default: True)
    overwrite_existing : bool
        Whether to overwrite existing keys (default: True)
    recursive_merge : bool
        Whether to recursively merge nested dictionaries (default: False)
    encryption_key : str, optional
        Key for encrypting/decrypting the file

    Returns:
    --------
    Path
        Path to the saved file
    """
    import json

    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    # Convert NumPy types if needed
    if convert_numpy:
        item = json_utils.convert_numpy_types(item)

    # Read existing data or create new dictionary
    current_data = {}
    if file_path.exists():
        try:
            # Try to read the existing file
            current_data = read_json(file_path, encoding, encryption_key)

            # Ensure it's a dictionary
            if not isinstance(current_data, dict):
                logger.warning(f"Existing file {file_path} does not contain a JSON object. Creating new object.")
                current_data = {}
        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Error reading existing JSON from {file_path}: {e}. Creating new object.")
            current_data = {}
    elif not create_if_missing:
        raise FileNotFoundError(f"File not found: {file_path} and create_if_missing is False")

    # Merge the dictionaries
    merged_data = json_utils.merge_json_objects_in_memory(
        current_data,
        item,
        overwrite_existing,
        recursive_merge
    )

    # Write back to the file
    return write_json(
        merged_data,
        file_path,
        encoding=encoding,
        indent=indent,
        convert_numpy=False,  # Already converted
        encryption_key=encryption_key
    )


# ====================
# Parquet Reading and Writing Functions
# ====================

def read_parquet(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """
    Reads a Parquet file into a DataFrame.

    Parameters:
    -----------
    file_path : str or Path
        Path to the Parquet file
    **kwargs
        Additional arguments to pass to pandas.read_parquet

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the file data
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading Parquet file: {file_path}")
    start_time = time.time()

    try:
        # Check if pyarrow is installed
        format_utils.check_pyarrow_available()

        df = pd.read_parquet(file_path, **kwargs)

        duration = time.time() - start_time
        logger.info(f"Read Parquet file {file_path} in {duration:.2f}s")
        return df
    except Exception as e:
        logger.error(f"Error reading Parquet file {file_path}: {e}")
        raise


def write_parquet(df: pd.DataFrame,
                  file_path: Union[str, Path],
                  compression: str = "snappy",
                  index: bool = False,
                  **kwargs) -> Path:
    """
    Writes a DataFrame to a Parquet file.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to write
    file_path : str or Path
        Path to save the Parquet file
    compression : str
        Compression algorithm (default: "snappy")
    index : bool
        Whether to include the index (default: False)
    **kwargs
        Additional arguments to pass to pd.DataFrame.to_parquet

    Returns:
    --------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    logger.info(f"Writing DataFrame to Parquet: {file_path}")
    start_time = time.time()

    try:
        # Check if pyarrow is installed
        format_utils.check_pyarrow_available()

        df.to_parquet(file_path, compression=compression, index=index, **kwargs)

        duration = time.time() - start_time
        logger.info(f"Wrote DataFrame to Parquet {file_path} in {duration:.2f}s")
        return file_path
    except Exception as e:
        logger.error(f"Error writing Parquet file {file_path}: {e}")
        raise


# ====================
# Image/Plot Utilities
# ====================

def save_visualization(figure: Any, file_path: Union[str, Path],
                       format: str = "png", **kwargs) -> Path:
    """
    Universal function for saving visualizations of different types.

    Parameters:
    -----------
    figure : Any
        Visualization object (Plotly, Matplotlib, PIL Image, WordCloud)
    file_path : Union[str, Path]
        Path to save the visualization
    format : str, optional
        Format (png, jpg, svg, html), default is "png"
    **kwargs :
        Additional parameters specific to the visualization type

    Returns:
    --------
    Path
        Path to the saved file

    Raises:
    -------
    TypeError
        If the visualization type is not supported
    """

    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    # Make sure the file extension matches the format
    if not str(file_path).lower().endswith(f".{format.lower()}"):
        file_path = file_path.with_suffix(f".{format.lower()}")

    logger.info(f"Saving visualization to {file_path}")

    try:
        # Handle Plotly figure
        if isinstance(figure, go.Figure):
            if format.lower() == 'html':
                figure.write_html(file_path, **kwargs)
            else:
                figure.write_image(str(file_path), format=format, **kwargs)

        # Handle Matplotlib figure
        elif hasattr(figure, 'savefig'):  # Matplotlib figure
            # Default to tight layout if not specified
            if 'bbox_inches' not in kwargs:
                kwargs['bbox_inches'] = 'tight'

            # Get DPI from kwargs or use default
            dpi = kwargs.pop('dpi', 300)

            figure.savefig(file_path, format=format, dpi=dpi, **kwargs)

        # Handle PIL Image
        elif isinstance(figure, Image.Image):
            # Get DPI from kwargs or use default
            dpi = kwargs.pop('dpi', 300)
            figure.save(file_path, format=format.upper(), dpi=(dpi, dpi), **kwargs)

        # Handle WordCloud result dictionary
        elif isinstance(figure, dict) and 'image' in figure and isinstance(figure['image'], Image.Image):
            # Get DPI from kwargs or use default
            dpi = kwargs.pop('dpi', 300)
            figure['image'].save(file_path, format=format.upper(), dpi=(dpi, dpi), **kwargs)

        else:
            raise TypeError(f"Unsupported visualization type: {type(figure)}")

        logger.info(f"Visualization saved to {file_path}")
        return file_path

    except Exception as e:
        logger.error(f"Error saving visualization to {file_path}: {e}")
        raise


def save_plot(plot_fig, file_path: Union[str, Path], dpi: int = 300, **kwargs) -> Path:
    """
    Saves a matplotlib or plotly figure to a file.

    Parameters:
    -----------
    plot_fig : matplotlib.figure.Figure or plotly.graph_objects.Figure
        The figure to save
    file_path : str or Path
        Path to save the image
    dpi : int
        Dots per inch for raster formats (default: 300)
    **kwargs
        Additional arguments to pass to the underlying save function

    Returns:
    --------
    Path
        Path to the saved file
    """
    # Determine format from file extension
    file_path = Path(file_path)
    file_format = file_path.suffix.lower().lstrip('.')

    # Use the universal save_visualization function with appropriate parameters
    return save_visualization(plot_fig, file_path, format=file_format, dpi=dpi, **kwargs)


# ====================
# Data Transformation Functions
# ====================

def save_dataframe(df: pd.DataFrame,
                   file_path: Union[str, Path],
                   format: str = "csv",
                   **kwargs) -> Path:
    """
    Saves a DataFrame to a file in the specified format.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to save
    file_path : str or Path
        Path to save the file
    format : str
        File format: "csv", "json", "parquet", "excel", "pickle" (default: "csv")
    **kwargs
        Additional arguments to pass to the underlying save function

    Returns:
    --------
    Path
        Path to the saved file
    """
    file_path = Path(file_path)

    # Ensure directory exists
    ensure_directory(file_path.parent)

    # Make sure the file extension matches the format
    if not str(file_path).lower().endswith(f".{format.lower()}"):
        file_path = file_path.with_suffix(f".{format.lower()}")

    logger.info(f"Saving DataFrame to {file_path} in {format} format ({len(df)} rows)")
    start_time = time.time()

    # Initialize result variable to avoid potential reference errors
    result = None

    # Save based on format
    if format.lower() == "csv":
        result = write_dataframe_to_csv(df, file_path, **kwargs)
    elif format.lower() == "json":
        # For JSON, we need to convert the DataFrame to a list or dict
        # Initialize data variable to satisfy static analyzers
        data = None

        # Get orient parameter with default value
        orient_value = kwargs.pop("orient", "records")

        # Explicitly handle each possible orient value to satisfy type checking
        if orient_value == "dict":
            data = df.to_dict(orient="dict")
        elif orient_value == "list":
            data = df.to_dict(orient="list")
        elif orient_value == "series":
            data = df.to_dict(orient="series")
        elif orient_value == "split":
            data = df.to_dict(orient="split")
        elif orient_value == "tight":
            data = df.to_dict(orient="tight")
        elif orient_value == "index":
            data = df.to_dict(orient="index")
        else:
            # Default to "records" for any other value
            if orient_value != "records":
                logger.warning(f"Invalid orient value: {orient_value}. Using 'records' instead.")
            data = df.to_dict(orient="records")

        # Safety check to ensure data is initialized
        if data is None:
            logger.warning("Unexpected condition: data was not initialized. Using default 'records' orient.")
            data = df.to_dict(orient="records")

        result = write_json(data, file_path, **kwargs)
    elif format.lower() == "parquet":
        result = write_parquet(df, file_path, **kwargs)
    elif format.lower() == "excel":
        # Check if openpyxl is available
        try:
            format_utils.check_openpyxl_available()
            df.to_excel(file_path, **kwargs)
            result = file_path
        except ImportError:
            logger.error("openpyxl is required to write Excel files")
            raise ImportError(
                "openpyxl is required to write Excel files. Please install it with 'pip install openpyxl'.")
    elif format.lower() == "pickle":
        df.to_pickle(file_path, **kwargs)
        result = file_path
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Verify result is initialized
    if result is None:
        raise RuntimeError(f"Failed to save DataFrame to {file_path}. No result was returned.")

    duration = time.time() - start_time
    logger.info(f"Saved DataFrame to {file_path} in {duration:.2f}s")

    return result


def read_dataframe(file_path: Union[str, Path],
                  file_format: Optional[str] = None,
                  **kwargs) -> pd.DataFrame:
    """
    Reads a file into a DataFrame based on the file extension or specified format.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file
    file_format : str, optional
        File format: "csv", "json", "parquet", "excel", "pickle".
        If None, inferred from file extension.
    **kwargs
        Additional arguments to pass to the underlying read function

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the file data
    """


    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Infer format from file extension if not specified
    if file_format is None:
        file_format = format_utils.detect_format_from_extension(file_path)

    logger.info(f"Reading file {file_path} as {file_format}")
    start_time = time.time()

    # Initialize DataFrame variable to avoid potential reference errors
    df = None

    # Read based on format
    if file_format.lower() == "csv":
        df = read_full_csv(file_path, **kwargs)
    elif file_format.lower() == "json":
        # For JSON, we need to specify the orient
        orient_value = kwargs.pop("orient", "records")
        json_data = read_json(file_path, **kwargs)

        # Handle each possible orient value explicitly to satisfy type checking
        if orient_value == "columns":
            df = pd.DataFrame.from_dict(json_data, orient="columns")
        elif orient_value == "index":
            df = pd.DataFrame.from_dict(json_data, orient="index")
        elif orient_value == "tight":
            df = pd.DataFrame.from_dict(json_data, orient="tight")
        else:
            # Default to "columns" for any other value
            if orient_value != "columns":
                logger.warning(f"Invalid orient value: {orient_value}. Using 'columns' instead.")
            df = pd.DataFrame.from_dict(json_data, orient="columns")
    elif file_format.lower() == "parquet":
        df = read_parquet(file_path, **kwargs)
    elif file_format.lower() == "excel":
        # Check if openpyxl is available
        try:
            format_utils.check_openpyxl_available()
            df = pd.read_excel(file_path, **kwargs)
        except ImportError:
            logger.error("openpyxl is required to read Excel files")
            raise ImportError(
                "openpyxl is required to read Excel files. Please install it with 'pip install openpyxl'.")
    elif file_format.lower() == "pickle":
        df = pd.read_pickle(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    # Safety check to ensure df is initialized
    if df is None:
        raise RuntimeError(f"Failed to read DataFrame from {file_path}. No data was returned.")

    duration = time.time() - start_time
    logger.info(f"Read {len(df)} rows from {file_path} in {duration:.2f}s")

    return df