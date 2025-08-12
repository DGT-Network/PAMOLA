"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Data Source Abstraction
Description: Unified data source interface for operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides a standardized abstraction for various data sources,
allowing operations to work with data from different origins through a
consistent interface regardless of the underlying storage format.

Key features:
- Unified access to DataFrames and file paths
- Schema validation and compatibility checking
- Multi-file dataset handling
- Encryption support through pamola core crypto utilities
- Memory optimization for large dataset handling
- Standardized error reporting

Implementation delegates file operations to the DataReader class while providing
a convenient abstraction layer for operational code.
"""

from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple, Generator, TypeVar

import dask.dataframe as dd
import pandas as pd

from pamola_core.utils import logging as custom_logging
from pamola_core.utils.ops import op_data_source_helpers
from pamola_core.utils.ops.op_data_reader import DataReader, ResultWithError

# Define type for file paths dictionary - allowing both Path and List[Path] as values
PathType = TypeVar('PathType', Path, List[Path])


class DataSource:
    """
    Represents a data source for operations.

    This class abstracts away the details of where the data comes from,
    allowing operations to work with data from various sources in a uniform way.
    """

    def __init__(self,
                 dataframes: Dict[str, Union[pd.DataFrame, dd.DataFrame]] = None,
                 file_paths: Dict[str, Union[Path, List[Path]]] = None,
                 encryption_keys: Dict[str, Union[str, Path]] = None,
                 encryption_modes: Dict[str, Union[str, Path]] = None):
        """
        Initialize a data source.

        Parameters:
        -----------
        dataframes : Dict[str, Union[pd.DataFrame, dd.DataFrame]], optional
            Dictionary of named DataFrames
        file_paths : Dict[str, Union[Path, List[Path]]], optional
            Dictionary of named file paths (can be single paths or lists of paths)
        """
        self.dataframes = dataframes or {}
        self.file_paths = file_paths or {}
        self.encryption_keys = encryption_keys or {}
        self.encryption_modes = encryption_modes or {}

        # Initialize logger using the custom logging module
        self.logger = custom_logging.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing DataSource")

        # Convert string paths to Path objects
        for key, path in self.file_paths.items():
            if isinstance(path, str):
                self.file_paths[key] = Path(path)
                self.logger.debug(f"Converted string path to Path object for '{key}'")
            elif isinstance(path, list):
                self.file_paths[key] = [Path(p) if isinstance(p, str) else p for p in path]
                self.logger.debug(f"Converted string paths to Path objects for list '{key}'")

        # Initialize DataReader for all file operations
        self.reader = DataReader(logger=self.logger)

        # Cache for schema information
        self._schema_cache = {}

        self.logger.debug(f"DataSource initialized with {len(self.dataframes)} dataframes and "
                          f"{len(self.file_paths)} file paths")

    def __enter__(self):
        """Support for context manager protocol."""
        self.logger.debug("Entering DataSource context")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        self.logger.debug("Exiting DataSource context")
        # Clear caches
        self._schema_cache.clear()

        if exc_type:
            self.logger.error(f"Exception during DataSource context: {exc_type.__name__}: {exc_val}")

        return False  # Don't suppress exceptions

    def add_dataframe(self, name: str, df: Union[pd.DataFrame, dd.DataFrame]):
        """
        Add a DataFrame to the data source.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame to add
        """
        self.dataframes[name] = df
        self.logger.debug(
            f"Added DataFrame '{name}'"
            f" with {len(df) if isinstance(df, pd.DataFrame) else int(df.map_partitions(len).sum().compute())} rows"
            f" and {len(df.columns)} columns"
        )

        # Clear cached schema for this dataframe
        if name in self._schema_cache:
            del self._schema_cache[name]
            self.logger.debug(f"Cleared schema cache for '{name}'")
            
    def add_encryption_key(self, name: str, encryption_key: Union[str, Path]):
        existed_encryption_key = self.encryption_keys.get(name)
        if not existed_encryption_key:
            self.encryption_keys[name] = encryption_key
            self.logger.debug(f"Added encryption_key '{name}': {encryption_key}")

            # Clear cached schema for this file
            if name in self._schema_cache:
                del self._schema_cache[name]
                self.logger.debug(f"Cleared schema cache for '{name}'")
        else:
            self.logger.debug(f"encryption_key '{name}' is existed.")
            
    def add_encryption_mode(self, name: str, path: Union[str, Path]):
        existed_encryption_mode = self.encryption_modes.get(name)
        if not existed_encryption_mode:
            from pamola_core.utils.io_helpers.crypto_utils import detect_encryption_mode
            encryption_mode = detect_encryption_mode(path)
            self.encryption_modes[name] = encryption_mode
            self.logger.debug(f"Added encryption_mode '{name}': {encryption_mode}")

            # Clear cached schema for this file
            if name in self._schema_cache:
                del self._schema_cache[name]
                self.logger.debug(f"Cleared schema cache for '{name}'")
        else:
            self.logger.debug(f"encryption_mode '{name}' is existed.")
            
    def add_file_path(self, name: str, path: Union[str, Path]):
        """
        Add a file path to the data source.

        Parameters:
        -----------
        name : str
            Name of the file
        path : Union[str, Path]
            Path to the file
        """
        path_obj = Path(path) if isinstance(path, str) else path
        self.file_paths[name] = path_obj
        self.logger.debug(f"Added file path '{name}': {path_obj}")

        # Clear cached schema for this file
        if name in self._schema_cache:
            del self._schema_cache[name]
            self.logger.debug(f"Cleared schema cache for '{name}'")

    def suggest_engine(
            self,
            name: str
    ) -> str:
        """
        Suggest engine should to use.

        Parameters:
        -----------
        name : str
            Name of the DataFrame

        Returns:
        --------
        str
            Engine should use
        """
        available_engines = ["dask", "pandas"]
        suggest_engine = "pandas"

        return suggest_engine

    def get_dataframe(
            self,
            name: str,
            load_if_path: bool = True,
            columns: Optional[List[str]] = None,
            nrows: Optional[int] = None,
            skiprows: Optional[Union[int, List[int]]] = None,
            encoding: str = "utf-8",
            delimiter: str = ",",
            quotechar: str = '"',
            use_dask: bool = False,
            memory_limit: Optional[float] = None,
            encryption_key: Optional[str] = None,
            show_progress: bool = True,
            validate_schema: Optional[Dict[str, Any]] = None,
            detect_parameters: bool = True,
            use_encryption: bool = False,
            encryption_mode: Optional[str] = None,
    ) -> ResultWithError:
        """
        Get a DataFrame by name with enhanced error reporting and schema validation.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        load_if_path : bool
            Whether to load from file if DataFrame is not in memory
        columns : List[str], optional
            Specific columns to load (reduces memory usage for wide datasets)
        nrows : int, optional
            Maximum number of rows to load (useful for sampling large datasets)
        skiprows : Union[int, List[int]], optional
            Row indices to skip or number of rows to skip from the start
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        use_dask : bool
            Whether to use Dask for distributed processing of large files
        memory_limit : float, optional
            Memory limit in GB for auto-switching to Dask
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress bars during loading
        validate_schema : Dict[str, Any], optional
            Schema to validate against after loading

        Returns:
        --------
        Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        # Try getting from memory first
        df, error_info = self._get_dataframe_from_memory(name, columns)
        if df is not None:
            # Validate schema if requested
            if validate_schema and df is not None:
                is_valid, validation_errors = self._validate_dataframe_schema(df, validate_schema)
                if not is_valid:
                    error_info = {
                        "error_type": "SchemaValidationError",
                        "message": f"DataFrame '{name}' failed schema validation",
                        "validation_errors": validation_errors
                    }
                    self.logger.warning(error_info["message"])
                    return df, error_info
            return df, error_info

        # Try loading from file if requested
        if load_if_path:
            df, error_info = self._get_dataframe_from_file(
                name, columns, nrows, skiprows, encoding, delimiter, quotechar,
                use_dask, memory_limit, encryption_key, show_progress, detect_parameters=detect_parameters,
                use_encryption=use_encryption, encryption_mode=encryption_mode
            )

            if df is not None:
                # Store in memory for future use
                self.dataframes[name] = df

                # Validate schema if requested
                if validate_schema:
                    is_valid, validation_errors = self._validate_dataframe_schema(df, validate_schema)
                    if not is_valid:
                        error_info = {
                            "error_type": "SchemaValidationError",
                            "message": f"DataFrame '{name}' failed schema validation",
                            "validation_errors": validation_errors
                        }
                        self.logger.warning(error_info["message"])
                        return df, error_info

                return df, None
            elif error_info:
                return None, error_info

        # Not found
        error_info = {
            "error_type": "DataFrameNotFoundError",
            "message": f"DataFrame '{name}' not found"
        }
        self.logger.debug(error_info["message"])
        return None, error_info

    def _validate_dataframe_schema(
            self, df: Union[pd.DataFrame, dd.DataFrame], expected_schema: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against an expected schema.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame to validate
        expected_schema : Dict[str, Any]
            Expected schema information

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        # Build actual schema from DataFrame
        actual_schema = self._build_schema_from_df(df)

        # Delegate schema validation to the helper function
        return op_data_source_helpers.validate_schema(
            actual_schema=actual_schema,
            expected_schema=expected_schema,
            logger=self.logger
        )

    def _build_schema_from_df(self, df: Union[pd.DataFrame, dd.DataFrame]) -> Dict[str, Any]:
        """
        Build schema information from DataFrame.

        Parameters:
        -----------
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame to build schema from

        Returns:
        --------
        Dict[str, Any]
            Schema information
        """
        # Build actual schema from DataFrame
        schema = {
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns} if isinstance(df, pd.DataFrame)
            else {col: str(df[col].map_partitions(lambda x: x.dtype).compute().iloc[0]) for col in df.columns},
            'num_rows': len(df) if isinstance(df, pd.DataFrame)
            else int(df.map_partitions(len).sum().compute()),
            'num_cols': len(df.columns),
            'null_counts': {col: int(df[col].isna().sum()) for col in df.columns} if isinstance(df, pd.DataFrame)
            else {col: int(df[col].isna().map_partitions(lambda part: part.sum()).sum().compute()) for col in df.columns}
        }

        # Try to get unique counts for columns (not for very large datasets)
        if schema.get("num_rows", 100000) < 100000:
            try:
                sample = df if isinstance(df, pd.DataFrame) else df.compute()
                schema['unique_counts'] = {col: int(sample[col].nunique()) for col in sample.columns}
            except Exception as e:
                self.logger.debug(f"Could not compute unique counts: {e}")

        return schema

    def _get_dataframe_from_memory(self, name: str,
                                   columns: Optional[List[str]] = None) -> ResultWithError:
        """
        Get a DataFrame from memory.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        columns : List[str], optional
            Specific columns to include

        Returns:
        --------
        Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        if name in self.dataframes:
            df = self.dataframes[name]

            # If columns specified, return only those columns
            if columns is not None:
                # Validate columns exist
                missing_cols = [col for col in columns if col not in df.columns]
                if missing_cols:
                    self.logger.warning(f"Columns not found in DataFrame '{name}': {missing_cols}")

                valid_cols = [col for col in columns if col in df.columns]
                if not valid_cols:
                    error_info = {
                        "error_type": "ColumnNotFoundError",
                        "message": f"None of the requested columns exist in DataFrame '{name}'",
                        "requested_columns": columns,
                        "available_columns": list(df.columns)
                    }
                    self.logger.error(error_info["message"])
                    return None, error_info

                return df[valid_cols], None

            return df, None

        return None, None

    def _get_dataframe_from_file(self, name: str,
                                 columns: Optional[List[str]] = None,
                                 nrows: Optional[int] = None,
                                 skiprows: Optional[Union[int, List[int]]] = None,
                                 encoding: str = "utf-8",
                                 delimiter: str = ",",
                                 quotechar: str = '"',
                                 use_dask: bool = False,
                                 memory_limit: Optional[float] = None,
                                 encryption_key: Optional[str] = None,
                                 show_progress: bool = True,
                                 auto_optimize: bool = True,
                                 detect_parameters: bool = True,
                                 use_encryption: bool = False,
                                 encryption_mode: Optional[str] = None) -> ResultWithError:
        """
        Load a DataFrame from a file using DataReader.

        Parameters:
        -----------
        name : str
            Name of the DataFrame/file
        columns : List[str], optional
            Specific columns to load
        nrows : int, optional
            Maximum number of rows to load
        skiprows : Union[int, List[int]], optional
            Row indices to skip or number of rows to skip from the start
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        use_dask : bool
            Whether to use Dask for distributed processing
        memory_limit : float, optional
            Memory limit in GB for auto-switching to Dask
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress bars during loading
        auto_optimize : bool
            Whether to optimize memory usage after loading

        Returns:
        --------
        Tuple[Optional[Union[pd.DataFrame, dd.DataFrame]], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        if name not in self.file_paths:
            return None, None

        file_path = self.file_paths[name]
        self.logger.debug(f"DataFrame '{name}' not in memory, attempting to load from file: {file_path}")

        # For a multi-file dataset (list of paths)
        if isinstance(file_path, list):
            try:
                # Create a dictionary with a single entry for multi-file dataset
                # This matches the expected type for DataReader.read_dataframe source parameter
                file_dict = {name: file_path}

                return self.reader.read_dataframe(
                    source=file_dict,  # Pass as dictionary
                    columns=columns,
                    nrows=nrows,
                    skiprows=skiprows,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    use_dask=use_dask,
                    memory_limit=memory_limit,
                    encryption_key=encryption_key,
                    auto_optimize=auto_optimize,
                    show_progress=show_progress
                )
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "resolution": "Check file format and ensure it is readable"
                }
                self.logger.error(f"Error loading multiple files: {error_info['error_type']}: {error_info['message']}")
                return None, error_info

        # For a single file path
        elif isinstance(file_path, Path) and file_path.exists():
            try:
                # For single files, we can use read_dataframe directly with the Path
                return self.reader.read_dataframe(
                    source=file_path,  # This is a single Path, which is a valid parameter
                    columns=columns,
                    nrows=nrows,
                    skiprows=skiprows,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    use_dask=use_dask,
                    memory_limit=memory_limit,
                    encryption_key=encryption_key,
                    auto_optimize=auto_optimize,
                    show_progress=show_progress,
                    detect_parameters=detect_parameters,
                    use_encryption=use_encryption,
                    encryption_mode=encryption_mode
                )
            except Exception as e:
                error_info = {
                    "error_type": type(e).__name__,
                    "message": str(e),
                    "file_path": str(file_path),
                    "resolution": "Check file format and ensure it is readable"
                }
                self.logger.error(f"Error loading file '{name}': {error_info['error_type']}: {error_info['message']}")
                return None, error_info
        else:
            error_info = {
                "error_type": "FileNotFoundError",
                "message": f"File path does not exist: {file_path}"
            }
            self.logger.warning(error_info["message"])
            return None, error_info

    def get_file_path(self, name: str) -> Optional[Path]:
        """
        Get a file path by name.

        Parameters:
        -----------
        name : str
            Name of the file

        Returns:
        --------
        Path or None
            The requested file path, or None if not found
        """
        if name in self.file_paths:
            self.logger.debug(f"Retrieved file path for '{name}'")
            return self.file_paths.get(name)

        self.logger.debug(f"File path '{name}' not found")
        return None

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get schema information for a DataFrame.

        Parameters:
        -----------
        name : str
            Name of the DataFrame

        Returns:
        --------
        Dict[str, Any] or None
            Schema information or None if DataFrame not found
        """
        # Check if schema is already cached
        if name in self._schema_cache:
            self.logger.debug(f"Retrieved schema for '{name}' from cache")
            return self._schema_cache[name]

        # Get the DataFrame
        df, error_info = self.get_dataframe(name)
        if df is None:
            self.logger.warning(f"Cannot get schema: DataFrame '{name}' not found: {error_info.get('message')}")
            return None

        # Build schema information
        self.logger.debug(f"Building schema for DataFrame '{name}'")
        schema = self._build_schema_from_df(df)

        # Add sample values if dataset is not too large
        if len(df) > 0 and len(df) <= 10000:
            try:
                sample_row = df.iloc[0].to_dict()
                schema['sample_values'] = {k: str(v) for k, v in sample_row.items()}
            except Exception as e:
                self.logger.debug(f"Could not extract sample values: {e}")

        # Cache the schema
        self._schema_cache[name] = schema
        self.logger.debug(f"Schema for '{name}' cached: {len(schema['columns'])} columns, {schema['num_rows']} rows")

        return schema

    def validate_schema(self, name: str, expected_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a DataFrame against an expected schema.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        expected_schema : Dict[str, Any]
            Expected schema information

        Returns:
        --------
        Tuple[bool, List[str]]
            (is_valid, error_messages)
        """
        self.logger.debug(f"Validating schema for '{name}'")

        # Validate expected_schema type
        if not isinstance(expected_schema, dict):
            self.logger.error("Expected schema must be a dictionary")
            return False, ["Expected schema must be a dictionary"]

        # Get the DataFrame schema
        schema = self.get_schema(name)
        if schema is None:
            self.logger.error(f"Cannot validate schema: DataFrame '{name}' not found")
            return False, ["DataFrame not found"]

        # Delegate schema validation to the helper function
        return op_data_source_helpers.validate_schema(schema, expected_schema, self.logger)

    def get_dataframe_chunks(self,
                             name: str,
                             chunk_size: int = 10000,
                             columns: Optional[List[str]] = None,
                             encoding: str = "utf-8",
                             delimiter: str = ",",
                             quotechar: str = '"',
                             encryption_key: Optional[str] = None,
                             show_progress: bool = True) -> Generator[pd.DataFrame, None, None]:
        """
        Get chunks of a DataFrame for processing large datasets.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        chunk_size : int
            Size of each chunk
        columns : List[str], optional
            Specific columns to include in the chunks (reduces memory usage)
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress during processing

        Yields:
        -------
        pd.DataFrame
            Chunks of the DataFrame
        """
        # Check if we have a DataFrame in memory
        if name in self.dataframes:
            self.logger.debug(f"Getting chunks from in-memory DataFrame '{name}'")
            df = self.dataframes[name]

            # Use helper function to generate chunks
            yield from op_data_source_helpers.generate_dataframe_chunks(
                df, chunk_size, columns, self.logger, show_progress
            )

        # Try to load from file
        elif name in self.file_paths:
            file_path = self.file_paths[name]

            # For a single file path
            if isinstance(file_path, Path) and file_path.exists():
                # Use DataReader to read the file in chunks
                try:
                    for chunk in self.reader.read_dataframe_in_chunks(
                            source=file_path,
                            chunk_size=chunk_size,
                            columns=columns,
                            encoding=encoding,
                            delimiter=delimiter,
                            quotechar=quotechar,
                            encryption_key=encryption_key,
                            show_progress=show_progress
                    ):
                        yield chunk
                except Exception as e:
                    self.logger.error(f"Error reading chunks from '{name}': {str(e)}")
            else:
                self.logger.error(f"Cannot generate chunks: File path does not exist or is not a single file")
        else:
            self.logger.error(f"Cannot generate chunks: DataFrame '{name}' not found")

    def add_multi_file_dataset(self,
                               name: str,
                               file_paths: List[Union[str, Path]],
                               load: bool = False,
                               columns: Optional[List[str]] = None,
                               min_valid_files: int = 1,
                               error_on_empty: bool = False,
                               encoding: str = "utf-8",
                               delimiter: str = ",",
                               quotechar: str = '"',
                               encryption_key: Optional[str] = None,
                               show_progress: bool = True):
        """
        Add a dataset consisting of multiple files.

        Parameters:
        -----------
        name : str
            Name for the dataset
        file_paths : List[Union[str, Path]]
            List of file paths
        load : bool
            Whether to load the dataset immediately
        columns : List[str], optional
            Specific columns to load (reduces memory usage)
        min_valid_files : int
            Minimum number of valid files required (raises error if fewer)
        error_on_empty : bool
            Whether to raise error if no valid files found or final dataset is empty
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress during processing
        """
        # Convert all paths to Path objects
        paths = [Path(p) if isinstance(p, str) else p for p in file_paths]
        self.logger.info(f"Adding multi-file dataset '{name}' with {len(paths)} files")

        # Store as a special entry in file_paths
        self.file_paths[name] = paths

        # If load is requested, we need to handle the multi-file dataset loading
        if load:
            try:
                # We use our internal _get_dataframe_from_file method which handles multi-file datasets
                df, error_info = self._get_dataframe_from_file(
                    name=name,
                    columns=columns,
                    nrows=None,
                    skiprows=None,
                    encoding=encoding,
                    delimiter=delimiter,
                    quotechar=quotechar,
                    encryption_key=encryption_key,
                    show_progress=show_progress,
                    auto_optimize=True
                )

                if df is not None:
                    self.dataframes[name] = df
                    self.logger.info(f"Loaded multi-file dataset '{name}' with {len(df)} rows")
                elif error_on_empty and error_info:
                    error_msg = error_info.get("message", "Unknown error loading dataset")
                    raise ValueError(error_msg)
            except Exception as e:
                if error_on_empty:
                    raise
                self.logger.error(f"Error loading multi-file dataset '{name}': {str(e)}")

    def release_dataframe(self, name: str) -> bool:
        """
        Release a DataFrame from memory to free up resources.

        This is useful when working with very large datasets where
        memory management is critical.

        Parameters:
        -----------
        name : str
            Name of the DataFrame to release

        Returns:
        --------
        bool
            True if the DataFrame was released, False if not found
        """
        if name in self.dataframes:
            self.logger.info(f"Releasing DataFrame '{name}' from memory")
            del self.dataframes[name]
            # Force garbage collection to ensure memory is freed
            import gc
            gc.collect()
            return True
        else:
            self.logger.debug(f"DataFrame '{name}' not found in memory, nothing to release")
            return False

    def get_task_encryption_key(self, task_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the encryption key and metadata for a task.

        This is a helper function that centralizes access to task encryption keys.

        Parameters:
        -----------
        task_id : str, optional
            ID of the task

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with encryption key and metadata or None if not available
        """
        if task_id is None:
            return None

        try:
            from pamola_core.utils.crypto_helpers.key_store import get_key_for_task
            key = get_key_for_task(task_id)
            if key:
                return {
                    "key": key,
                    "mode": "simple",  # Default mode
                    "task_id": task_id
                }
            return None
        except ImportError:
            # Fallback if crypto module is not available
            self.logger.warning("Crypto key store module not available")
            return None

    def get_file_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive metadata about a file.

        Parameters:
        -----------
        name : str
            Name of the file in the data source

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with file metadata or None if file not found
        """
        file_path = self.get_file_path(name)
        if file_path is None or not isinstance(file_path, Path) or not file_path.exists():
            return None

        try:
            # Delegate to DataReader for format detection
            format_info = self.reader.detect_file_format(file_path)
            return format_info
        except Exception as e:
            self.logger.error(f"Error getting file metadata: {str(e)}")
            return None

    def get_encryption_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get encryption information for a file.

        Parameters:
        -----------
        name : str
            Name of the file

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with encryption information or None if file not encrypted
        """
        file_path = self.get_file_path(name)
        if file_path is None or not isinstance(file_path, Path) or not file_path.exists():
            return None

        try:
            # Delegate to DataReader
            return self.reader.get_encryption_info(file_path)
        except Exception as e:
            self.logger.warning(f"Failed to get encryption info: {str(e)}")
            return None

    def estimate_memory_usage(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Estimate memory requirements for loading a dataset.

        Parameters:
        -----------
        name : str
            Name of the DataFrame or file in the data source

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with memory requirement estimates
        """
        # For in-memory DataFrame
        if name in self.dataframes:
            df = self.dataframes[name]
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
            return {
                "source": "memory",
                "current_memory_mb": memory_usage,
                "estimated_memory_mb": memory_usage,
                "already_loaded": True
            }

        # For file path
        file_path = self.get_file_path(name)
        if file_path is None or not isinstance(file_path, Path) or not file_path.exists():
            return None

        try:
            # Delegate to DataReader
            return self.reader.estimate_memory_usage(file_path)
        except Exception as e:
            self.logger.error(f"Error estimating memory usage: {str(e)}")
            return None

    def optimize_memory(self, threshold_percent: float = 80.0) -> Dict[str, Any]:
        """
        Optimize memory usage by releasing and optimizing DataFrames.

        Parameters:
        -----------
        threshold_percent : float
            Memory usage threshold to trigger optimization (default: 80%)

        Returns:
        --------
        Dict[str, Any]
            Optimization results
        """
        return op_data_source_helpers.optimize_memory_usage(
            self.dataframes, threshold_percent, self.release_dataframe, self.logger
        )

    def analyze_dataframe(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Perform comprehensive analysis of a DataFrame.

        Parameters:
        -----------
        name : str
            Name of the DataFrame

        Returns:
        --------
        Dict[str, Any] or None
            Analysis results or None if DataFrame not found
        """
        df, error_info = self.get_dataframe(name)
        if df is None:
            self.logger.warning(f"Cannot analyze DataFrame '{name}': {error_info.get('message')}")
            return None

        return op_data_source_helpers.analyze_dataframe(df, self.logger)

    def create_sample(self, name: str, sample_size: int = 1000, random_seed: int = 42) -> ResultWithError:
        """
        Create a representative sample of a DataFrame.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        sample_size : int
            Number of rows in the sample
        random_seed : int
            Random seed for reproducibility

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        df, error_info = self.get_dataframe(name)
        if df is None:
            return None, error_info

        try:
            sample_df = op_data_source_helpers.create_sample_dataframe(
                df, sample_size, random_seed, True, self.logger
            )
            return sample_df, None
        except Exception as e:
            error_info = {
                "error_type": type(e).__name__,
                "message": str(e)
            }
            self.logger.error(f"Error creating sample: {error_info['message']}")
            return None, error_info

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "main"):
        """
        Create a DataSource from a single DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to use
        name : str
            Name to use for the DataFrame

        Returns:
        --------
        DataSource
            DataSource containing the DataFrame
        """
        data_source = cls(dataframes={name: df})
        data_source.logger.info(f"Created DataSource from DataFrame '{name}' with {len(df)} rows")
        return data_source

    @classmethod
    def from_file_path(cls, path: Union[str, Path], name: str = "main", load: bool = False):
        """
        Create a DataSource from a file path.

        Parameters:
        -----------
        path : Union[str, Path]
            Path to the file
        name : str
            Name to use for the file
        load : bool
            Whether to load the DataFrame immediately

        Returns:
        --------
        DataSource
            DataSource containing the file path (and DataFrame if load=True)
        """
        path_obj = Path(path) if isinstance(path, str) else path
        logger = custom_logging.get_logger(f"{__name__}.DataSource")
        logger.info(f"Creating DataSource from file path: {path_obj}")

        # Create a new DataSource
        data_source = cls(file_paths={name: path_obj})

        # Load the DataFrame if requested
        if load:
            logger.debug(f"Loading DataFrame from file path: {path_obj}")
            data_source.get_dataframe(name)

        return data_source

    @classmethod
    def from_multi_file_dataset(cls,
                                paths: List[Union[str, Path]],
                                name: str = "main",
                                load: bool = False,
                                encoding: str = "utf-8",
                                delimiter: str = ",",
                                quotechar: str = '"',
                                encryption_key: Optional[str] = None,
                                show_progress: bool = True):
        """
        Create a DataSource from multiple files.

        Parameters:
        -----------
        paths : List[Union[str, Path]]
            List of file paths
        name : str
            Name to use for the dataset
        load : bool
            Whether to load the dataset immediately
        encoding : str
            File encoding for text-based formats (default: "utf-8")
        delimiter : str
            Field delimiter for CSV files (default: ",")
        quotechar : str
            Text qualifier character for CSV files (default: '"')
        encryption_key : str, optional
            Key for decrypting encrypted files
        show_progress : bool
            Whether to show progress during processing

        Returns:
        --------
        DataSource
            DataSource containing the file paths (and DataFrame if load=True)
        """
        logger = custom_logging.get_logger(f"{__name__}.DataSource")
        logger.info(f"Creating DataSource from {len(paths)} files for dataset '{name}'")

        # Create a new DataSource
        data_source = cls()

        # Add the multi-file dataset
        data_source.add_multi_file_dataset(
            name=name,
            file_paths=paths,
            load=load,
            encoding=encoding,
            delimiter=delimiter,
            quotechar=quotechar,
            encryption_key=encryption_key,
            show_progress=show_progress
        )

        return data_source