"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Multi-file Dataset Utilities
Description: Helpers for processing, stacking, and merging multi-file datasets
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

Key features:
- Vertical stacking of CSV datasets across multiple files
- Batch processing for large multi-file collections
- Integration with progress tracking and memory management
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Callable, Tuple

import pandas as pd

from pamola_core.utils import progress
# Remove the direct import of functions from io.py to avoid circular imports
# from pamola_core.utils.io import (read_full_csv, read_parquet, read_excel, read_json, read_text)
from pamola_core.utils.io_helpers import error_utils
from pamola_core.utils.io_helpers import format_utils
from pamola_core.utils.io_helpers import memory_utils

# Configure module logger
logger = logging.getLogger("pamola_core.utils.io_helpers.multi_file_utils")


def detect_file_format(file_path: Union[str, Path]) -> str:
    """
    Detect file format from file extension.

    Parameters:
    -----------
    file_path : str or Path
        Path to the file

    Returns:
    --------
    str
        Detected format name
    """
    return format_utils.detect_format_from_extension(file_path)


def get_file_reader(file_format: str) -> Callable:
    """
    Get the appropriate file reader function for a format.

    Parameters:
    -----------
    file_format : str
        Format name (e.g., 'csv', 'parquet')

    Returns:
    --------
    Callable
        Reader function for the format
    """
    # Import the io module functions at the function level to avoid circular imports
    from pamola_core.utils.io import (
        read_full_csv, read_parquet, read_excel, read_json, read_text
    )

    format_readers = {
        "csv": read_full_csv,
        "parquet": read_parquet,
        "excel": read_excel,
        "json": read_json,
        "text": read_text,
        "txt": read_text,
    }

    if file_format.lower() in format_readers:
        return format_readers[file_format.lower()]
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def validate_files_exist(file_paths: List[Union[str, Path]]) -> Tuple[List[Path], List[str]]:
    """
    Validate that all specified files exist.

    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of file paths to validate

    Returns:
    --------
    Tuple[List[Path], List[str]]
        (valid_paths, missing_paths)
    """
    valid_paths = []
    missing_paths = []

    for file_path in file_paths:
        path = Path(file_path)
        if path.exists() and path.is_file():
            valid_paths.append(path)
        else:
            missing_paths.append(str(path))

    return valid_paths, missing_paths


def get_common_columns(file_paths: List[Union[str, Path]],
                       sample_rows: int = 5,
                       require_all_files: bool = False) -> List[str]:
    """
    Identify columns common to all files in the dataset.

    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of file paths to check
    sample_rows : int
        Number of rows to sample for column detection
    require_all_files : bool
        Whether to require all files to be valid

    Returns:
    --------
    List[str]
        List of common column names
    """
    # Validate files first
    valid_paths, missing_paths = validate_files_exist(file_paths)

    if not valid_paths:
        logger.error("No valid files found")
        return []

    if missing_paths and require_all_files:
        logger.error(f"Missing files: {missing_paths}")
        return []

    # Initialize common columns with first file
    try:
        first_file = valid_paths[0]
        file_format = detect_file_format(first_file)
        reader = get_file_reader(file_format)

        # Read sample to get columns
        if file_format in ["csv", "parquet", "txt", "text"]:
            df_sample = reader(first_file, nrows=sample_rows)
        else:
            # Excel, JSON don't support nrows parameter
            df_sample = reader(first_file)
            if len(df_sample) > sample_rows:
                df_sample = df_sample.head(sample_rows)

        common_columns = set(df_sample.columns)
        logger.debug(f"Initial columns from {first_file.name}: {len(common_columns)}")

        # Check each remaining file
        for file_path in valid_paths[1:]:
            try:
                file_format = detect_file_format(file_path)
                reader = get_file_reader(file_format)

                # Read sample
                if file_format in ["csv", "parquet", "txt", "text"]:
                    df_sample = reader(file_path, nrows=sample_rows)
                else:
                    df_sample = reader(file_path)
                    if len(df_sample) > sample_rows:
                        df_sample = df_sample.head(sample_rows)

                # Update common columns
                file_columns = set(df_sample.columns)
                common_columns = common_columns.intersection(file_columns)

                logger.debug(f"Columns after {file_path.name}: {len(common_columns)}")

                if not common_columns:
                    logger.warning("No common columns found across files")
                    break

            except Exception as e:
                logger.warning(f"Error reading columns from {file_path}: {str(e)}")
                if require_all_files:
                    logger.error("Required file could not be read, aborting")
                    return []

        return sorted(list(common_columns))

    except Exception as e:
        logger.error(f"Error finding common columns: {str(e)}")
        return []


def stack_files_vertically(
        file_paths: List[Union[str, Path]],
        columns: Optional[List[str]] = None,
        ignore_errors: bool = False,
        show_progress: bool = True,
        **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Stack multiple files vertically (row-wise) with error handling.

    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of file paths to stack
    columns : List[str], optional
        Specific columns to include (must be present in all files)
    ignore_errors : bool
        Whether to ignore errors in individual files
    show_progress : bool
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to reader functions

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        Combined DataFrame or error information
    """
    # Validate files first
    valid_paths, missing_paths = validate_files_exist(file_paths)

    if not valid_paths:
        return error_utils.create_error_info(
            "NoValidFilesError",
            "No valid files found",
            f"Check that files exist and are accessible: {missing_paths}"
        )

    if missing_paths and not ignore_errors:
        return error_utils.create_error_info(
            "MissingFilesError",
            f"Missing files: {missing_paths}",
            "Use ignore_errors=True to proceed with valid files only"
        )

    # Determine common columns if not specified
    if columns is None:
        common_cols = get_common_columns(valid_paths)
        if not common_cols:
            return error_utils.create_error_info(
                "NoCommonColumnsError",
                "No common columns found across files",
                "Specify columns parameter with column names to use"
            )
        columns = common_cols
        logger.info(f"Using {len(columns)} common columns found across files")

    # Initialize tracking variables
    dataframes = []
    error_files = []
    empty_files = []
    total_rows = 0

    # Create progress bar if requested
    progress_bar = None
    if show_progress:
        progress_bar = progress.ProgressBar(
            total=len(valid_paths),
            description="Reading files",
            unit="files"
        )

    # Process each file
    start_time = time.time()
    for file_path in valid_paths:
        try:
            file_format = detect_file_format(file_path)
            reader = get_file_reader(file_format)

            # Read file with appropriate parameters
            file_kwargs = kwargs.copy()

            # Add format-specific parameters
            if file_format in ["csv", "parquet", "txt", "text"]:
                if "columns" not in file_kwargs and columns is not None:
                    if file_format in ["csv", "parquet"]:
                        file_kwargs["usecols"] = columns

            # Read file
            df = reader(file_path, **file_kwargs)

            # Filter columns if needed and not already done by reader
            if columns is not None and not set(columns).issubset(set(df.columns)):
                # Find columns that exist in this file
                valid_cols = [col for col in columns if col in df.columns]
                if not valid_cols:
                    logger.warning(f"No requested columns found in {file_path}")
                    if show_progress:
                        progress_bar.update(1, postfix={"status": "no columns"})
                    continue

                df = df[valid_cols]

            # Skip empty dataframes
            if df.empty:
                logger.warning(f"File {file_path} contains no data")
                empty_files.append(str(file_path))
                if show_progress:
                    progress_bar.update(1, postfix={"status": "empty"})
                continue

            # Add to list of dataframes
            dataframes.append(df)
            total_rows += len(df)

            if show_progress:
                progress_bar.update(1, postfix={
                    "files": len(dataframes),
                    "rows": total_rows
                })

        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {str(e)}")
            error_files.append({
                "file": str(file_path),
                "error": str(e)
            })

            if not ignore_errors and len(error_files) > 0:
                if show_progress:
                    progress_bar.close()
                return error_utils.create_error_info(
                    "ReadError",
                    f"Error reading file: {str(e)}",
                    "Use ignore_errors=True to skip files with errors",
                    details={"file_path": str(file_path), "errors": error_files}
                )

            if show_progress:
                progress_bar.update(1, postfix={"status": "error"})

    if show_progress:
        progress_bar.close()

    # Check if we have any valid dataframes
    if not dataframes:
        return error_utils.create_error_info(
            "NoValidDataError",
            "No valid data found in files",
            "Check file contents and format",
            details={
                "error_files": error_files,
                "empty_files": empty_files
            }
        )

    # Combine dataframes
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Calculate metrics
        duration = time.time() - start_time

        logger.info(
            f"Combined {len(dataframes)} files with {total_rows} total rows "
            f"in {duration:.2f}s"
        )

        return combined_df

    except Exception as e:
        return error_utils.create_error_info(
            "CombineError",
            f"Error combining dataframes: {str(e)}",
            "Check that files have compatible structures",
            details={
                "valid_files": len(dataframes),
                "error_files": len(error_files),
                "empty_files": len(empty_files)
            }
        )


def process_files_in_batches(
        file_paths: List[Union[str, Path]],
        batch_size: int = 5,
        columns: Optional[List[str]] = None,
        ignore_errors: bool = False,
        show_progress: bool = True,
        processor: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Process multiple files in memory-efficient batches.

    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of file paths to process
    batch_size : int
        Number of files to process in each batch
    columns : List[str], optional
        Specific columns to include
    ignore_errors : bool
        Whether to ignore errors in individual files
    show_progress : bool
        Whether to show a progress bar
    processor : Callable, optional
        Function to apply to each batch before combining
    **kwargs
        Additional arguments to pass to reader functions

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        Combined DataFrame or error information
    """
    # Validate files first
    valid_paths, missing_paths = validate_files_exist(file_paths)

    if not valid_paths:
        return error_utils.create_error_info(
            "NoValidFilesError",
            "No valid files found",
            f"Check that files exist and are accessible: {missing_paths}"
        )

    if missing_paths and not ignore_errors:
        return error_utils.create_error_info(
            "MissingFilesError",
            f"Missing files: {missing_paths}",
            "Use ignore_errors=True to proceed with valid files only"
        )

    # Calculate number of batches
    num_batches = (len(valid_paths) + batch_size - 1) // batch_size

    # Initialize tracking variables
    batch_results = []
    error_files = []
    empty_files = []
    total_rows = 0

    # Create progress bar if requested
    progress_bar = None
    if show_progress:
        progress_bar = progress.ProgressBar(
            total=num_batches,
            description="Processing batches",
            unit="batches"
        )

    # Process batches
    start_time = time.time()
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(valid_paths))
        batch_paths = valid_paths[batch_start:batch_end]

        logger.info(f"Processing batch {batch_idx + 1}/{num_batches} with {len(batch_paths)} files")

        # Process this batch of files
        batch_result = stack_files_vertically(
            batch_paths,
            columns=columns,
            ignore_errors=ignore_errors,
            show_progress=False,  # Don't show nested progress bars
            **kwargs
        )

        # Check if result is an error
        if error_utils.is_error_info(batch_result):
            if not ignore_errors:
                if show_progress:
                    progress_bar.close()
                return batch_result
            else:
                # Track errors but continue
                logger.warning(f"Errors in batch {batch_idx + 1}: {batch_result['message']}")
                if 'details' in batch_result and 'errors' in batch_result['details']:
                    error_files.extend(batch_result['details']['errors'])
                if 'details' in batch_result and 'empty_files' in batch_result['details']:
                    empty_files.extend(batch_result['details']['empty_files'])

                if show_progress:
                    progress_bar.update(1, postfix={"status": "error"})

                continue

        # Process batch result if requested
        if processor is not None:
            try:
                batch_result = processor(batch_result)
            except Exception as e:
                if not ignore_errors:
                    if show_progress:
                        progress_bar.close()
                    return error_utils.create_error_info(
                        "ProcessorError",
                        f"Error in batch processor: {str(e)}",
                        "Check the processor function for errors"
                    )
                else:
                    logger.warning(f"Processor error in batch {batch_idx + 1}: {str(e)}")
                    if show_progress:
                        progress_bar.update(1, postfix={"status": "processor error"})
                    continue

        # Add batch result and update metrics
        batch_rows = len(batch_result)
        total_rows += batch_rows
        batch_results.append(batch_result)

        # Update progress
        if show_progress:
            progress_bar.update(1, postfix={
                "batches": len(batch_results),
                "rows": total_rows
            })

        # Optional garbage collection after each batch
        if batch_idx < num_batches - 1:  # Not the last batch
            try:
                import gc
                gc.collect()
            except:
                pass

    if show_progress:
        progress_bar.close()

    # Check if we have any valid results
    if not batch_results:
        return error_utils.create_error_info(
            "NoValidDataError",
            "No valid data found in files",
            "Check file contents and format",
            details={
                "error_files": error_files,
                "empty_files": empty_files
            }
        )

    # Combine all batch results
    try:
        combined_df = pd.concat(batch_results, ignore_index=True)

        # Calculate metrics
        duration = time.time() - start_time

        logger.info(
            f"Processed {len(valid_paths)} files in {num_batches} batches "
            f"with {total_rows} total rows in {duration:.2f}s"
        )

        return combined_df

    except Exception as e:
        return error_utils.create_error_info(
            "CombineError",
            f"Error combining batch results: {str(e)}",
            "Check that all batches have compatible structures"
        )


def read_multi_csv(
        file_paths: List[Union[str, Path]],
        encoding: str = "utf-8",
        delimiter: str = ",",
        quotechar: str = '"',
        columns: Optional[List[str]] = None,
        nrows: Optional[int] = None,
        skiprows: Optional[Union[int, List[int]]] = None,
        ignore_errors: bool = False,
        show_progress: bool = True,
        memory_efficient: bool = True,
        encryption_key: Optional[str] = None
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Read multiple CSV files and combine them vertically.

    Parameters:
    -----------
    file_paths : List[Union[str, Path]]
        List of CSV file paths to read
    encoding : str
        File encoding (default: "utf-8")
    delimiter : str
        Field delimiter (default: ",")
    quotechar : str
        Text qualifier character (default: '"')
    columns : List[str], optional
        Specific columns to read
    nrows : int, optional
        Maximum number of rows to read from each file
    skiprows : Union[int, List[int]], optional
        Rows to skip in each file
    ignore_errors : bool
        Whether to ignore errors in individual files
    show_progress : bool
        Whether to show a progress bar
    memory_efficient : bool
        Whether to use memory-efficient processing
    encryption_key : str, optional
        Key for decrypting encrypted files

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        Combined DataFrame or error information
    """
    # Get system memory info for intelligent batching
    system_memory = memory_utils.get_system_memory()
    available_memory_gb = system_memory.get("available_gb", 4.0)

    # For small datasets or when memory efficiency is not needed, use simple stacking
    if not memory_efficient or len(file_paths) < 3:
        return stack_files_vertically(
            file_paths,
            columns=columns,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            encoding=encoding,
            delimiter=delimiter,
            quotechar=quotechar,
            nrows=nrows,
            skiprows=skiprows,
            encryption_key=encryption_key
        )

    # For larger datasets, use batch processing
    # Calculate batch size based on available memory
    memory_per_file_gb = 0.25  # Rough estimate: 250MB per file

    # Adjust based on file sizes if possible
    try:
        # Sample a few files to get better size estimate
        sample_files = file_paths[:min(3, len(file_paths))]

        # Get average file size
        total_size_mb = 0
        valid_samples = 0

        for file_path in sample_files:
            try:
                size_estimate = memory_utils.estimate_csv_size(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter
                )

                if "estimated_memory_mb" in size_estimate:
                    total_size_mb += size_estimate["estimated_memory_mb"]
                    valid_samples += 1
            except:
                pass

        if valid_samples > 0:
            avg_memory_mb = total_size_mb / valid_samples
            memory_per_file_gb = avg_memory_mb / 1024

            logger.debug(f"Estimated memory per file: {memory_per_file_gb:.2f} GB")
    except:
        # Fallback to default estimate
        pass

    # Calculate batch size with safety margin (use 40% of available memory)
    max_batch_memory_gb = available_memory_gb * 0.4
    batch_size = max(1, int(max_batch_memory_gb / memory_per_file_gb))

    logger.info(f"Using batch size of {batch_size} files based on memory estimate")

    # Process in batches
    return process_files_in_batches(
        file_paths,
        batch_size=batch_size,
        columns=columns,
        ignore_errors=ignore_errors,
        show_progress=show_progress,
        encoding=encoding,
        delimiter=delimiter,
        quotechar=quotechar,
        nrows=nrows,
        skiprows=skiprows,
        encryption_key=encryption_key
    )


def read_similar_files(
        directory: Union[str, Path],
        pattern: str = "*.csv",
        recursive: bool = False,
        columns: Optional[List[str]] = None,
        ignore_errors: bool = False,
        show_progress: bool = True,
        **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Read multiple similar files from a directory.

    Parameters:
    -----------
    directory : str or Path
        Directory to search for files
    pattern : str
        Glob pattern for matching files (default: "*.csv")
    recursive : bool
        Whether to search recursively in subdirectories
    columns : List[str], optional
        Specific columns to read
    ignore_errors : bool
        Whether to ignore errors in individual files
    show_progress : bool
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to reader functions

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        Combined DataFrame or error information
    """
    directory = Path(directory)

    if not directory.exists() or not directory.is_dir():
        return error_utils.create_error_info(
            "DirectoryNotFoundError",
            f"Directory not found: {directory}",
            "Check the directory path"
        )

    # Find matching files
    file_paths = []
    if recursive:
        file_paths = list(directory.glob(f"**/{pattern}"))
    else:
        file_paths = list(directory.glob(pattern))

    if not file_paths:
        return error_utils.create_error_info(
            "NoFilesFoundError",
            f"No files matching pattern '{pattern}' found in {directory}",
            "Check the pattern or directory"
        )

    logger.info(f"Found {len(file_paths)} files matching pattern '{pattern}'")

    # Extract format from the first file for format-specific processing
    file_format = detect_file_format(file_paths[0])

    # Use appropriate multi-file reader based on format
    if file_format.lower() == "csv":
        return read_multi_csv(
            file_paths,
            columns=columns,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            **kwargs
        )
    else:
        # For other formats, use generic vertical stacking
        return stack_files_vertically(
            file_paths,
            columns=columns,
            ignore_errors=ignore_errors,
            show_progress=show_progress,
            **kwargs
        )


def memory_efficient_processor(
        processor: Callable[[pd.DataFrame], pd.DataFrame],
        file_paths: List[Union[str, Path]],
        batch_size: int = 5,
        columns: Optional[List[str]] = None,
        show_progress: bool = True,
        **kwargs
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Apply a processor function to files in a memory-efficient way.

    Parameters:
    -----------
    processor : Callable
        Function to apply to each batch of data
    file_paths : List[Union[str, Path]]
        List of file paths to process
    batch_size : int
        Number of files to process in each batch
    columns : List[str], optional
        Specific columns to include
    show_progress : bool
        Whether to show a progress bar
    **kwargs
        Additional arguments to pass to reader functions

    Returns:
    --------
    Union[pd.DataFrame, Dict[str, Any]]
        Processed DataFrame or error information
    """
    return process_files_in_batches(
        file_paths,
        batch_size=batch_size,
        columns=columns,
        show_progress=show_progress,
        processor=processor,
        **kwargs
    )