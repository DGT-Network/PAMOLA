"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Data Reader
Description: Unified data reading capabilities for operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides a specialized reader for various data sources,
ensuring optimal integration with io.py and consistent handling of
progress tracking, error reporting, and memory management.

Key features:
- Unified reading interface for all supported file formats
- Transparent encryption/decryption support
- Memory-optimized loading with pre-flight checks
- Automatic format detection and parameter inference
- Comprehensive progress tracking and error handling
- Multi-file dataset handling with memory-efficient processing
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple, Generator, TypeVar

import pandas as pd

from pamola_core.utils import logging as custom_logging
from pamola_core.utils.io import (
    # Pamola Core reading functions
    read_full_csv, read_csv_in_chunks, read_parquet, read_excel,
    read_json, read_dataframe, read_multi_csv, read_similar_files,
    # Memory management
    estimate_file_memory, optimize_dataframe_memory,
    # Format detection
    detect_csv_dialect, validate_file_format, is_encrypted_file,
    # Get file info/metadata
    get_file_metadata
)
from pamola_core.utils.progress import HierarchicalProgressTracker

# Define type variables for better type hints
PathType = Union[str, Path]
DataFrameType = TypeVar('DataFrameType', bound=pd.DataFrame)
ResultWithError = Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]


class DataReader:
    """
    Specialized reader for various data sources with optimal io.py integration.

    This class provides a consistent interface for reading data from various
    sources with robust error handling, progress tracking, and memory management.
    """

    def __init__(self, logger=None):
        """
        Initialize a DataReader instance.

        Parameters:
        -----------
        logger : logging.Logger, optional
            Logger instance for output messages
        """
        # Initialize logger
        self.logger = logger or custom_logging.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing DataReader")

    def read_dataframe(self,
                       source: Union[PathType, Dict[str, PathType]],
                       file_format: Optional[str] = None,
                       columns: Optional[List[str]] = None,
                       nrows: Optional[int] = None,
                       skiprows: Optional[Union[int, List[int]]] = None,
                       encoding: str = "utf-8",
                       delimiter: str = ",",
                       quotechar: str = '"',
                       sheet_name: Optional[Union[str, int]] = 0,
                       use_dask: bool = False,
                       memory_limit: Optional[float] = None,
                       encryption_key: Optional[str] = None,
                       auto_optimize: bool = True,
                       show_progress: bool = True,
                       detect_parameters: bool = True,
                       **kwargs) -> ResultWithError:
        """
        Read a DataFrame from various sources with comprehensive options.

        This method provides a unified interface for reading data from different
        sources with robust error handling, progress tracking, and memory management.

        Parameters:
        -----------
        source : str, Path, or Dict[str, Union[str, Path]]
            Source file path or dictionary of source paths
        file_format : str, optional
            File format override (auto-detected if None)
        columns : List[str], optional
            Specific columns to read (reduces memory usage)
        nrows : int, optional
            Maximum number of rows to read
        skiprows : Union[int, List[int]], optional
            Rows to skip when reading
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        sheet_name : str or int, optional
            Sheet name/index for Excel files (default: 0)
        use_dask : bool
            Whether to use Dask for distributed processing
        memory_limit : float, optional
            Memory limit in GB for auto-switching to Dask
        encryption_key : str, optional
            Key for decrypting encrypted files
        auto_optimize : bool
            Whether to optimize memory usage after loading
        show_progress : bool
            Whether to show progress bars
        detect_parameters : bool
            Whether to auto-detect format and parameters
        **kwargs
            Additional arguments passed to the underlying reader

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        # Initialize result and error info
        df = None
        error_info = None
        memory_info = {}  # Инициализируем переменную memory_info

        # If source is a dictionary, assume it's a multi-file dataset
        if isinstance(source, dict) or (isinstance(source, list) and len(source) > 0):
            return self._read_multi_file_dataset(
                source,
                columns=columns,
                nrows=nrows,
                skiprows=skiprows,
                encoding=encoding,
                delimiter=delimiter,
                quotechar=quotechar,
                encryption_key=encryption_key,
                show_progress=show_progress,
                auto_optimize=auto_optimize,
                **kwargs
            )

        # Convert source to Path if it's a string
        if isinstance(source, str):
            source = Path(source)

        # Validate source exists
        if not source.exists():
            error_info = {
                "error_type": "FileNotFoundError",
                "message": f"File not found: {source}",
                "resolution": "Check file path and permissions"
            }
            self.logger.error(error_info["message"])
            return None, error_info

        # Auto-detect file format if not specified
        if file_format is None:
            file_format = source.suffix.lower().lstrip('.')
            self.logger.debug(f"Auto-detected format: {file_format}")

        # Perform pre-flight memory check with io.py
        try:
            memory_info = estimate_file_memory(source)
            memory_required_gb = memory_info.get('estimated_memory_mb', 0) / 1024

            # Auto-switch to Dask if memory limit is specified and exceeded
            if memory_limit and memory_required_gb > memory_limit:
                self.logger.warning(
                    f"File {source} requires {memory_required_gb:.2f}GB RAM, "
                    f"exceeding limit of {memory_limit:.2f}GB. Switching to Dask."
                )
                use_dask = True
        except Exception as e:
            self.logger.warning(f"Memory estimation failed: {e}. Proceeding without pre-flight check.")

        # Auto-detect encoding and parameters if requested for CSV files
        if detect_parameters and file_format == 'csv':
            try:
                dialect_info = detect_csv_dialect(source)
                if dialect_info.get("encoding"):
                    encoding = dialect_info["encoding"]
                    self.logger.debug(f"Auto-detected encoding: {encoding}")
                if dialect_info.get("delimiter"):
                    delimiter = dialect_info["delimiter"]
                    self.logger.debug(f"Auto-detected delimiter: {delimiter}")
                if dialect_info.get("quotechar"):
                    quotechar = dialect_info["quotechar"]
                    self.logger.debug(f"Auto-detected quotechar: {quotechar}")
            except Exception as e:
                self.logger.warning(f"Auto-detection failed: {e}. Using specified parameters.")

        # Create progress tracker if requested
        progress_tracker = None
        if show_progress:
            total_estimate = nrows if nrows is not None else memory_info.get('estimated_rows', 1000)
            progress_tracker = HierarchicalProgressTracker(
                total=total_estimate,
                description=f"Reading {source.name}",
                unit="rows",
                track_memory=True
            )

        try:
            # Select appropriate reader based on format
            if file_format in ('csv', 'tsv', 'txt'):
                df, error_info = self._read_csv_format(
                    source, file_format, encoding, delimiter, quotechar,
                    columns, nrows, skiprows, use_dask,
                    encryption_key, show_progress,
                    progress_tracker, **kwargs
                )
            elif file_format in ('parquet', 'pq'):
                df, error_info = self._read_parquet_format(
                    source, columns, nrows, skiprows,
                    encryption_key, **kwargs
                )
            elif file_format in ('xls', 'xlsx', 'xlsm'):
                df, error_info = self._read_excel_format(
                    source, sheet_name, columns, nrows, skiprows,
                    encryption_key, show_progress, **kwargs
                )
            elif file_format == 'json':
                df, error_info = self._read_json_format(
                    source, encoding, columns, nrows, skiprows,
                    encryption_key, **kwargs
                )
            else:
                # For other formats, use generic read_dataframe from io.py
                try:
                    df = read_dataframe(
                        file_path=source,
                        file_format=file_format,
                        columns=columns,
                        nrows=nrows,
                        skiprows=skiprows,
                        encryption_key=encryption_key,
                        **kwargs
                    )
                except Exception as e:
                    error_info = {
                        "error_type": type(e).__name__,
                        "message": str(e),
                        "resolution": self._suggest_resolution(e, file_format)
                    }

            # Auto-optimize memory usage if requested and data was loaded
            if df is not None and auto_optimize:
                try:
                    df, optim_info = optimize_dataframe_memory(df, inplace=True)
                    self.logger.debug(
                        f"Memory optimization: {optim_info.get('savings_percent', 0):.1f}% saved"
                    )
                except Exception as e:
                    self.logger.warning(f"Memory optimization failed: {e}")

            # Update progress tracker
            if progress_tracker:
                if df is not None:
                    progress_tracker.update(len(df), {
                        "status": "complete",
                        "rows": len(df)
                    })
                else:
                    progress_tracker.update(0, {"status": "error"})
                progress_tracker.close()

            if df is not None:
                self.logger.info(f"Successfully loaded {len(df)} rows from {source}")
            else:
                self.logger.warning(f"No data loaded from {source}")

            return df, error_info

        except Exception as e:
            # Handle errors
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "source": str(source),
                "format": file_format,
                "resolution": self._suggest_resolution(e, file_format)
            }

            # Close progress tracker on error
            if progress_tracker:
                progress_tracker.update(0, {"status": "error", "message": str(e)})
                progress_tracker.close()

            self.logger.error(f"Error reading {source}: {error_info['error_type']}: {error_info['message']}")
            return None, error_info

    def _read_csv_format(self, source, file_format, encoding, delimiter, quotechar,
                         columns, nrows, skiprows, use_dask,
                         encryption_key, show_progress,
                         progress_tracker=None, **kwargs) -> ResultWithError:
        """Read data from CSV-like formats using io.py functions."""
        try:
            # For TSV format, override delimiter
            if file_format == 'tsv':
                delimiter = '\t'

            # Use io.py's read_full_csv
            df = read_full_csv(
                file_path=source,
                encoding=encoding,
                delimiter=delimiter,
                quotechar=quotechar,
                columns=columns,
                nrows=nrows,
                skiprows=skiprows,
                use_dask=use_dask,
                encryption_key=encryption_key,
                show_progress=show_progress
            )
            return df, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "resolution": self._suggest_resolution(e, file_format)
            }
            return None, error_info

    def _read_parquet_format(self, source, columns, nrows, skiprows,
                             encryption_key, **kwargs) -> ResultWithError:
        """Read data from Parquet format using io.py functions."""
        try:
            # Use io.py's read_parquet
            df = read_parquet(
                file_path=source,
                columns=columns,
                encryption_key=encryption_key,
                **kwargs
            )

            # Handle nrows and skiprows manually (not supported by read_parquet)
            if skiprows is not None:
                if isinstance(skiprows, int):
                    df = df.iloc[skiprows:]
                else:
                    keep_mask = ~pd.Series(range(len(df))).isin(skiprows)
                    df = df.loc[keep_mask].reset_index(drop=True)

            if nrows is not None:
                df = df.head(nrows)

            return df, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "resolution": self._suggest_resolution(e, "parquet")
            }
            return None, error_info

    def _read_excel_format(self, source, sheet_name, columns, nrows, skiprows,
                           encryption_key, show_progress, **kwargs) -> ResultWithError:
        """Read data from Excel format using io.py functions."""
        try:
            # Use io.py's read_excel
            df = read_excel(
                file_path=source,
                sheet_name=sheet_name,
                columns=columns,
                nrows=nrows,
                skiprows=skiprows,
                encryption_key=encryption_key,
                show_progress=show_progress,
                **kwargs
            )
            return df, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "resolution": self._suggest_resolution(e, "excel")
            }
            return None, error_info

    def _read_json_format(self, source, encoding, columns, nrows, skiprows,
                          encryption_key, **kwargs) -> ResultWithError:
        """Read data from JSON format using io.py functions."""
        try:
            # Use io.py's read_json
            data = read_json(
                file_path=source,
                encoding=encoding,
                encryption_key=encryption_key,
                **kwargs
            )

            # Convert to DataFrame based on structure
            if isinstance(data, list):
                # Handle list of records or values
                if data and isinstance(data[0], dict):
                    # List of dictionaries -> records orient
                    df = pd.DataFrame.from_records(data)
                else:
                    # List of values -> try to convert directly
                    df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to determine the JSON orientation
                if all(key in data for key in ['index', 'columns', 'data']):
                    # Full split orientation (3 keys)
                    df = pd.DataFrame(
                        data=data['data'],
                        index=data['index'],
                        columns=data['columns']
                    )
                elif 'columns' in data and 'data' in data:
                    # Simplified split orientation (2 keys)
                    df = pd.DataFrame(
                        data=data['data'],
                        columns=data['columns']
                    )
                elif 'schema' in data and 'data' in data:
                    # Table orientation (Table Schema format)
                    df = pd.DataFrame.from_records(data['data'])
                elif all(isinstance(v, list) for v in data.values()):
                    # Dict of lists -> columns oriented
                    df = pd.DataFrame(data)
                elif all(isinstance(v, dict) for v in data.values()):
                    # Dict of dicts -> could be index or columns oriented
                    # Try both orientations
                    try:
                        # First try index orientation
                        df = pd.DataFrame.from_dict(data, orient='index')
                    except (ValueError, TypeError):
                        try:
                            # Then try columns orientation
                            df = pd.DataFrame.from_dict(data, orient='columns')
                        except (ValueError, TypeError):
                            # If both fail, raise error
                            raise ValueError("Cannot determine orientation of dictionary data")
                else:
                    # Single record -> wrap in list
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"JSON data structure not convertible to DataFrame: {type(data)}")

            # Apply filtering
            if columns is not None:
                valid_cols = [col for col in columns if col in df.columns]
                if valid_cols:
                    df = df[valid_cols]
                else:
                    self.logger.warning(f"None of the requested columns exist in JSON data")

            if skiprows is not None:
                if isinstance(skiprows, int):
                    df = df.iloc[skiprows:]
                else:
                    keep_mask = ~pd.Series(range(len(df))).isin(skiprows)
                    df = df.loc[keep_mask].reset_index(drop=True)

            if nrows is not None:
                df = df.head(nrows)

            # Add warning for potential orientation issues
            if df is not None and df.empty:
                self.logger.warning("DataFrame is empty after conversion. Check if JSON orientation is correct.")
            elif df is not None and len(df.columns) == 1 and len(df) == 1:
                self.logger.warning("DataFrame has only one row and column. Check if JSON orientation is correct.")

            return df, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "resolution": self._suggest_resolution(e, "json")
            }
            return None, error_info

    def read_dataframe_in_chunks(self,
                                 source: PathType,
                                 chunk_size: int = 10000,
                                 columns: Optional[List[str]] = None,
                                 encoding: str = "utf-8",
                                 delimiter: str = ",",
                                 quotechar: str = '"',
                                 encryption_key: Optional[str] = None,
                                 show_progress: bool = True,
                                 detect_parameters: bool = True,
                                 **kwargs) -> Generator[pd.DataFrame, None, None]:
        """
        Read a file in chunks for memory-efficient processing of large datasets.

        Parameters:
        -----------
        source : PathType
            Source file path
        chunk_size : int
            Size of each chunk (default: 10000)
        columns : List[str], optional
            Specific columns to read
        encoding : str
            File encoding (default: "utf-8")
        delimiter : str
            Field delimiter (default: ",")
        quotechar : str
            Text qualifier character (default: '"')
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress during processing
        detect_parameters : bool
            Whether to auto-detect format and parameters
        **kwargs
            Additional arguments to pass to reader functions

        Yields:
        -------
        pd.DataFrame
            DataFrame chunks
        """
        # Convert source to Path if it's a string
        if isinstance(source, str):
            source = Path(source)

        # Validate source exists
        if not source.exists():
            self.logger.error(f"File not found: {source}")
            return

        # Get file format from extension
        file_format = source.suffix.lower().lstrip('.')

        # Auto-detect encoding and parameters if requested
        if detect_parameters and file_format == 'csv':
            try:
                dialect_info = detect_csv_dialect(source)
                if dialect_info.get("encoding"):
                    encoding = dialect_info["encoding"]
                    self.logger.debug(f"Auto-detected encoding: {encoding}")
                if dialect_info.get("delimiter"):
                    delimiter = dialect_info["delimiter"]
                    self.logger.debug(f"Auto-detected delimiter: {delimiter}")
                if dialect_info.get("quotechar"):
                    quotechar = dialect_info["quotechar"]
                    self.logger.debug(f"Auto-detected quotechar: {quotechar}")
            except Exception as e:
                self.logger.warning(f"Auto-detection failed: {e}. Using specified parameters.")

        # Process based on file format
        self.logger.info(f"Reading {source} in chunks of {chunk_size} rows")

        if file_format in ('csv', 'tsv', 'txt'):
            # For TSV format, override delimiter
            if file_format == 'tsv':
                delimiter = '\t'

            # Use io.py's read_csv_in_chunks directly
            try:
                for chunk in read_csv_in_chunks(
                        file_path=source,
                        chunk_size=chunk_size,
                        encoding=encoding,
                        delimiter=delimiter,
                        quotechar=quotechar,
                        columns=columns,
                        encryption_key=encryption_key,
                        show_progress=show_progress
                ):
                    yield chunk
            except Exception as e:
                self.logger.error(f"Error reading chunks from {source}: {str(e)}")
        else:
            # For other formats, load full file and yield chunks
            self.logger.warning(
                f"Chunked reading not directly supported for {file_format} format. "
                f"Loading full file and chunking in memory."
            )

            try:
                # Load full DataFrame
                df, error_info = self.read_dataframe(
                    source=source,
                    columns=columns,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    encryption_key=encryption_key,
                    show_progress=show_progress,
                    detect_parameters=detect_parameters,
                    **kwargs
                )

                if df is None:
                    if error_info:
                        self.logger.error(f"Failed to load {source}: {error_info.get('message', 'Unknown error')}")
                    return

                # Calculate total chunks properly
                total_chunks = (len(df) + chunk_size - 1) // chunk_size

                # Create progress tracker
                progress_tracker = None
                if show_progress:
                    progress_tracker = HierarchicalProgressTracker(
                        total=total_chunks,
                        description=f"Processing {source.name} in chunks",
                        unit="chunks",
                        track_memory=True
                    )

                # Yield chunks
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i + chunk_size].copy()

                    # Update progress without trying to get memory info
                    if progress_tracker:
                        progress_tracker.update(1, {
                            "chunk": f"{(i // chunk_size) + 1}/{total_chunks}",
                            "rows": len(chunk)
                        })

                    yield chunk

                # Close progress tracker
                if progress_tracker:
                    progress_tracker.close()

            except Exception as e:
                self.logger.error(f"Error chunking {source}: {str(e)}")

    def _read_multi_file_dataset(self,
                                 sources: Union[List[PathType], Dict[str, PathType]],
                                 columns: Optional[List[str]] = None,
                                 nrows: Optional[int] = None,
                                 skiprows: Optional[Union[int, List[int]]] = None,
                                 encoding: str = "utf-8",
                                 delimiter: str = ",",
                                 quotechar: str = '"',
                                 encryption_key: Optional[str] = None,
                                 min_valid_files: int = 1,
                                 memory_efficient: bool = True,
                                 auto_optimize: bool = True,
                                 show_progress: bool = True,
                                 error_on_empty: bool = False,
                                 **kwargs) -> ResultWithError:
        """
        Read and combine multiple files into a single DataFrame.

        This method fully leverages io.py's multi-file reading capabilities.

        Parameters:
        -----------
        sources : Union[List[PathType], Dict[str, PathType]]
            List or dictionary of source paths
        columns : List[str], optional
            Specific columns to read
        nrows : int, optional
            Maximum number of rows to read
        skiprows : Union[int, List[int]], optional
            Rows to skip when reading
        encoding : str
            File encoding (default: "utf-8")
        delimiter : str
            Field delimiter (default: ",")
        quotechar : str
            Text qualifier character (default: '"')
        encryption_key : str, optional
            Key for decrypting encrypted files
        min_valid_files : int
            Minimum number of valid files required
        memory_efficient : bool
            Whether to use memory-efficient processing
        auto_optimize : bool
            Whether to optimize memory usage after loading
        show_progress : bool
            Whether to show progress during processing
        error_on_empty : bool
            Whether to raise error if no data is loaded
        **kwargs
            Additional arguments to pass to reader functions

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        # Convert dictionary to list if needed
        if isinstance(sources, dict):
            sources = list(sources.values())

        # Convert all sources to Path objects
        paths = [Path(p) if isinstance(p, str) else p for p in sources]

        # Check if we have any valid paths
        valid_paths = [p for p in paths if p.exists()]

        if len(valid_paths) < min_valid_files:
            error_info = {
                "error_type": "InsufficientValidFilesError",
                "message": f"Found only {len(valid_paths)} valid files, but {min_valid_files} required",
                "resolution": "Check file paths and permissions"
            }
            self.logger.error(error_info["message"])

            if error_on_empty:
                raise ValueError(error_info["message"])

            return None, error_info

        # Get file format of the first file
        file_format = valid_paths[0].suffix.lower().lstrip('.')
        self.logger.info(f"Reading multi-file dataset with {len(valid_paths)} files of {file_format} format")

        try:
            # For CSV files, directly use read_multi_csv from io.py
            if file_format in ('csv', 'tsv', 'txt'):
                # For TSV format, override delimiter
                if file_format == 'tsv':
                    delimiter = '\t'

                # Use read_multi_csv without **kwargs
                df = read_multi_csv(
                    file_paths=valid_paths,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    columns=columns,
                    nrows=nrows,
                    skiprows=skiprows,
                    encryption_key=encryption_key,
                    show_progress=show_progress,
                    memory_efficient=memory_efficient,
                    ignore_errors=not error_on_empty
                )
            else:
                # For other formats, check if all files are in same directory
                base_dirs = {p.parent for p in valid_paths}

                if len(base_dirs) == 1:
                    # All files in same directory, use read_similar_files
                    base_dir = list(base_dirs)[0]
                    pattern = f"*{valid_paths[0].suffix.lower()}"

                    # Use read_similar_files without **kwargs
                    df = read_similar_files(
                        directory=base_dir,
                        pattern=pattern,
                        columns=columns,
                        encryption_key=encryption_key,
                        show_progress=show_progress,
                        ignore_errors=not error_on_empty
                    )
                else:
                    # Files in different directories, read individually and combine
                    self.logger.info(
                        f"Files in multiple directories. Reading individually and combining."
                    )

                    # Create progress tracker
                    progress_tracker = None
                    if show_progress:
                        progress_tracker = HierarchicalProgressTracker(
                            total=len(valid_paths),
                            description="Reading multiple files",
                            unit="files",
                            track_memory=True
                        )

                    dfs = []
                    for i, path in enumerate(valid_paths):
                        try:
                            file_df, _ = self.read_dataframe(
                                source=path,
                                columns=columns,
                                nrows=nrows,
                                skiprows=skiprows,
                                encoding=encoding,
                                delimiter=delimiter,
                                quotechar=quotechar,
                                encryption_key=encryption_key,
                                show_progress=False,  # Don't show progress for individual files
                                auto_optimize=False  # Optimize after combining
                            )

                            if file_df is not None and len(file_df) > 0:
                                dfs.append(file_df)

                            # Update progress
                            if progress_tracker:
                                progress_tracker.update(1, {
                                    "file": f"{i + 1}/{len(valid_paths)}",
                                    "successful": len(dfs)
                                })

                        except Exception as e:
                            self.logger.warning(f"Error reading {path}: {str(e)}")
                            if error_on_empty:
                                raise

                    # Close progress tracker
                    if progress_tracker:
                        progress_tracker.close()

                    # Combine DataFrames
                    if dfs:
                        df = pd.concat(dfs, ignore_index=True)
                    else:
                        df = None

            # Check if we have any data
            if df is None or len(df) == 0:
                error_info = {
                    "error_type": "EmptyDatasetError",
                    "message": f"No data loaded from {len(valid_paths)} files",
                    "resolution": "Check file contents or adjust parameters"
                }
                self.logger.warning(error_info["message"])

                if error_on_empty:
                    raise ValueError(error_info["message"])

                return None, error_info

            # Optimize memory usage if requested
            if auto_optimize:
                try:
                    df, optim_info = optimize_dataframe_memory(df, inplace=True)
                    self.logger.debug(
                        f"Memory optimization: {optim_info.get('savings_percent', 0):.1f}% saved"
                    )
                except Exception as e:
                    self.logger.warning(f"Memory optimization failed: {e}")

            self.logger.info(f"Successfully loaded {len(df)} rows from {len(valid_paths)} files")
            return df, None

        except Exception as e:
            # Handle errors
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e),
                "sources": [str(p) for p in valid_paths[:5]] + (["..."] if len(valid_paths) > 5 else []),
                "resolution": "Check file formats and contents"
            }
            self.logger.error(f"Error reading multiple files: {error_info['error_type']}: {error_info['message']}")
            return None, error_info

    def get_encryption_info(self, source: PathType) -> Optional[Dict[str, Any]]:
        """
        Get encryption information for a file using io.py functions.

        Parameters:
        -----------
        source : str or Path
            Source file path

        Returns:
        --------
        Dict[str, Any] or None
            Encryption information or None if file not encrypted
        """
        # Convert source to Path if it's a string
        if isinstance(source, str):
            source = Path(source)

        # Check if file exists
        if not source.exists():
            self.logger.warning(f"File not found: {source}")
            return None

        try:
            # Check if file is encrypted and get metadata if it is
            if is_encrypted_file(source):
                return get_file_metadata(source)
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get encryption info: {str(e)}")
            return None

    def detect_file_format(self, source: PathType) -> Optional[Dict[str, Any]]:
        """
        Detect format information for a file using io.py functions.

        Parameters:
        -----------
        source : str or Path
            Source file path

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with format information or None if detection fails
        """
        # Convert source to Path if it's a string
        if isinstance(source, str):
            source = Path(source)

        # Check if file exists
        if not source.exists():
            self.logger.warning(f"File not found: {source}")
            return None

        try:
            # Get file format from extension
            file_format = source.suffix.lower().lstrip('.')

            # Use io.py's detection functions
            if file_format in ('csv', 'tsv', 'txt'):
                return detect_csv_dialect(source)
            else:
                return validate_file_format(source)
        except Exception as e:
            self.logger.warning(f"Failed to detect file format: {str(e)}")
            return {
                "error": str(e),
                "format": source.suffix.lower().lstrip('.')
            }

    def estimate_memory_usage(self, source: PathType) -> Optional[Dict[str, Any]]:
        """
        Estimate memory requirements for loading a file using io.py.

        Parameters:
        -----------
        source : str or Path
            Source file path

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with memory requirement estimates
        """
        # Convert source to Path if it's a string
        if isinstance(source, str):
            source = Path(source)

        # Check if file exists
        if not source.exists():
            self.logger.warning(f"File not found: {source}")
            return None

        try:
            # Delegate to io.py's estimate_file_memory
            return estimate_file_memory(source)
        except Exception as e:
            self.logger.warning(f"Failed to estimate memory usage: {str(e)}")
            return {
                "error": str(e),
                "file_size_mb": source.stat().st_size / (1024 * 1024)
            }

    def _suggest_resolution(self, error: Exception, file_format: str) -> str:
        """
        Suggest resolution for common errors.

        Parameters:
        -----------
        error : Exception
            The exception that occurred
        file_format : str
            The file format being read

        Returns:
        --------
        str
            Suggested resolution
        """
        error_type = type(error).__name__
        error_message = str(error).lower()

        # CSV/Text file errors
        if file_format in ('csv', 'tsv', 'txt'):
            if 'unicode' in error_message or 'codec' in error_message:
                return "Try a different encoding (e.g., 'utf-8', 'latin1', 'cp1252')"
            elif 'delimiter' in error_message:
                return "Check or auto-detect the correct delimiter"
            elif 'quote' in error_message:
                return "Check or auto-detect the correct quote character"

        # Parquet errors
        elif file_format in ('parquet', 'pq'):
            if 'pyarrow' in error_message:
                return "Install pyarrow with 'pip install pyarrow'"
            elif 'corrupt' in error_message:
                return "File may be corrupted or incomplete"

        # Excel errors
        elif file_format in ('xls', 'xlsx', 'xlsm'):
            if 'openpyxl' in error_message:
                return "Install openpyxl with 'pip install openpyxl'"
            elif 'sheet' in error_message:
                return "Check sheet name or index"

        # JSON errors
        elif file_format == 'json':
            if 'json' in error_message and 'decode' in error_message:
                return "File may not be valid JSON format"

        # Encryption errors
        if 'encrypt' in error_message or 'decrypt' in error_message or 'key' in error_message:
            return "Check encryption key or file format"

        # Memory errors
        if error_type == 'MemoryError' or 'memory' in error_message:
            return "File too large for available memory. Try reading in chunks or use Dask."

        # Default resolution
        return "Check file format, encoding, and parameters"