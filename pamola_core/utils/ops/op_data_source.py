"""
Data source abstractions for operations in the HHR project.

This module provides classes for representing and handling different types of
data sources that operations can work with, including DataFrames and files.
"""

import warnings
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple, Generator, TypeVar

import pandas as pd

from pamola_core.utils.io import read_full_csv

# Define type for file paths dictionary - allowing both Path and List[Path] as values
PathType = TypeVar('PathType', Path, List[Path])

class DataSource:
    """
    Represents a data source for operations.

    This class abstracts away the details of where the data comes from,
    allowing operations to work with data from various sources in a uniform way.
    """

    def __init__(self,
                 dataframes: Dict[str, pd.DataFrame] = None,
                 file_paths: Dict[str, Union[Path, List[Path]]] = None):
        """
        Initialize a data source.

        Parameters:
        -----------
        dataframes : Dict[str, pd.DataFrame], optional
            Dictionary of named DataFrames
        file_paths : Dict[str, Union[Path, List[Path]]], optional
            Dictionary of named file paths (can be single paths or lists of paths)
        """
        self.dataframes = dataframes or {}
        self.file_paths = file_paths or {}

        # Convert string paths to Path objects
        for key, path in self.file_paths.items():
            if isinstance(path, str):
                self.file_paths[key] = Path(path)
            elif isinstance(path, list):
                self.file_paths[key] = [Path(p) if isinstance(p, str) else p for p in path]

        # Cache for schema information
        self._schema_cache = {}

        # Cache for data chunks
        self._chunk_cache = {}

    def add_dataframe(self, name: str, df: pd.DataFrame):
        """
        Add a DataFrame to the data source.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        df : pd.DataFrame
            DataFrame to add
        """
        self.dataframes[name] = df
        # Clear cached schema for this dataframe
        if name in self._schema_cache:
            del self._schema_cache[name]

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
        self.file_paths[name] = Path(path) if isinstance(path, str) else path
        # Clear cached schema for this file
        if name in self._schema_cache:
            del self._schema_cache[name]

    def get_dataframe(self, name: str, load_if_path: bool = True) -> Optional[pd.DataFrame]:
        """
        Get a DataFrame by name.

        If the DataFrame is not in memory but a file path with the same name
        exists, it will be loaded (if load_if_path is True).

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        load_if_path : bool
            Whether to load from file if DataFrame is not in memory

        Returns:
        --------
        pd.DataFrame or None
            The requested DataFrame, or None if not found
        """
        # First check if DataFrame is already in memory
        if name in self.dataframes:
            return self.dataframes[name]

        # If not, check if file path exists and should be loaded
        if load_if_path and name in self.file_paths:
            file_path = self.file_paths[name]
            if file_path.exists():
                # Determine file format based on extension
                ext = file_path.suffix.lower()

                try:
                    if ext == '.csv':
                        # Load DataFrame from CSV file
                        df = read_full_csv(file_path)
                    elif ext == '.parquet':
                        # Load DataFrame from Parquet file
                        df = pd.read_parquet(file_path)
                    elif ext in ['.xls', '.xlsx']:
                        # Load DataFrame from Excel file
                        df = pd.read_excel(file_path)
                    elif ext == '.json':
                        # Load DataFrame from JSON file
                        df = pd.read_json(file_path)
                    elif ext == '.txt':
                        # Load DataFrame from text file (assuming tab-delimited)
                        df = pd.read_csv(file_path, sep='\t')
                    else:
                        # Unsupported file format
                        return None

                    # Store in memory for future use
                    self.dataframes[name] = df
                    return df
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error loading file {file_path}: {str(e)}")
                    return None

        return None

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
        return self.file_paths.get(name)

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
            return self._schema_cache[name]

        # Get the DataFrame
        df = self.get_dataframe(name)
        if df is None:
            return None

        # Build schema information
        schema = {
            'columns': list(df.columns),
            'dtypes': {col: str(df[col].dtype) for col in df.columns},
            'num_rows': len(df),
            'num_cols': len(df.columns)
        }

        # Cache the schema
        self._schema_cache[name] = schema

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
        # Get the DataFrame schema
        schema = self.get_schema(name)
        if schema is None:
            return False, ["DataFrame not found"]

        errors = []

        # Check if all expected columns exist
        if 'columns' in expected_schema:
            missing_cols = set(expected_schema['columns']) - set(schema['columns'])
            if missing_cols:
                errors.append(f"Missing columns: {', '.join(missing_cols)}")

        # Check if column types match
        if 'dtypes' in expected_schema:
            for col, dtype in expected_schema['dtypes'].items():
                if col in schema['dtypes']:
                    actual_dtype = schema['dtypes'][col]
                    if not self._is_compatible_dtype(actual_dtype, dtype):
                        errors.append(f"Column '{col}' has type '{actual_dtype}', expected '{dtype}'")

        return len(errors) == 0, errors

    def _is_compatible_dtype(self, actual: str, expected: str) -> bool:
        """
        Check if two data types are compatible.

        Parameters:
        -----------
        actual : str
            Actual data type
        expected : str
            Expected data type

        Returns:
        --------
        bool
            True if compatible, False otherwise
        """
        # Convert string representations to standard forms
        actual = actual.lower()
        expected = expected.lower()

        # Check for exact match
        if actual == expected:
            return True

        # Check for compatible numeric types
        if ('int' in actual or 'float' in actual) and ('int' in expected or 'float' in expected):
            return True

        # Check for compatible string types
        if ('object' in actual or 'string' in actual) and ('object' in expected or 'string' in expected):
            return True

        # Check for compatible datetime types
        if 'datetime' in actual and 'datetime' in expected:
            return True

        return False

    def get_dataframe_chunks(self,
                             name: str,
                             chunk_size: int = 10000,
                             columns: Optional[List[str]] = None) -> Generator[pd.DataFrame, None, None]:
        """
        Get chunks of a DataFrame for processing large datasets.

        Parameters:
        -----------
        name : str
            Name of the DataFrame
        chunk_size : int
            Size of each chunk
        columns : List[str], optional
            Specific columns to include in the chunks

        Yields:
        -------
        pd.DataFrame
            Chunks of the DataFrame
        """
        # Get the DataFrame
        df = self.get_dataframe(name)
        if df is None:
            return

        # Select specific columns if requested
        if columns is not None:
            df = df[columns]

        # Determine the number of chunks
        num_chunks = (len(df) + chunk_size - 1) // chunk_size

        # Yield chunks
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            yield df.iloc[start_idx:end_idx].copy()

    def add_multi_file_dataset(self,
                               name: str,
                               file_paths: List[Union[str, Path]],
                               load: bool = False):
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
        """
        # Convert all paths to Path objects
        paths = [Path(p) if isinstance(p, str) else p for p in file_paths]

        # Store as a special entry in file_paths
        self.file_paths[name] = paths  # This is now allowed by the type annotation

        if load:
            # Load and combine all files
            dfs = []
            for path in paths:
                if not path.exists():
                    continue

                # Determine file format based on extension
                ext = path.suffix.lower()
                try:
                    if ext == '.csv':
                        df = pd.read_csv(path)
                    elif ext == '.parquet':
                        df = pd.read_parquet(path)
                    elif ext in ['.xls', '.xlsx']:
                        df = pd.read_excel(path)
                    else:
                        continue
                    dfs.append(df)
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error loading file {path}: {str(e)}")
                    continue

            if dfs:
                # Combine all dataframes
                combined_df = pd.concat(dfs, ignore_index=True)
                self.dataframes[name] = combined_df

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
        return cls(dataframes={name: df})

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

        if load:
            # Determine file format based on extension
            ext = path_obj.suffix.lower()

            try:
                if ext == '.csv':
                    df = pd.read_csv(path_obj)
                elif ext == '.parquet':
                    df = pd.read_parquet(path_obj)
                elif ext in ['.xls', '.xlsx']:
                    df = pd.read_excel(path_obj)
                elif ext == '.json':
                    df = pd.read_json(path_obj)
                elif ext == '.txt':
                    df = pd.read_csv(path_obj, sep='\t')
                else:
                    # If format not recognized, try CSV
                    warnings.warn(f"File format not recognized for {path_obj}, trying CSV")
                    df = pd.read_csv(path_obj)

                return cls(dataframes={name: df}, file_paths={name: path_obj})
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error loading file {path_obj}: {str(e)}")
                return cls(file_paths={name: path_obj})
        else:
            return cls(file_paths={name: path_obj})

    @classmethod
    def from_multi_file_dataset(cls,
                                paths: List[Union[str, Path]],
                                name: str = "main",
                                load: bool = False):
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

        Returns:
        --------
        DataSource
            DataSource containing the file paths (and DataFrame if load=True)
        """
        # Create a new DataSource
        data_source = cls()

        # Add the multi-file dataset
        data_source.add_multi_file_dataset(name, paths, load)

        return data_source