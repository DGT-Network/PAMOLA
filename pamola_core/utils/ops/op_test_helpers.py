"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Testing Helpers
Description: Test utilities for operation unit testing
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides testing utilities for PAMOLA.CORE operations,
including test doubles, assertion helpers, and fixture setup methods.
These utilities simplify unit testing of operations by removing
dependencies on real file I/O, encryption, and external resources.
"""

import json
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, TypeVar, NamedTuple

import pandas as pd

from pamola_core.utils import logging as custom_logging
from pamola_core.utils.io import ensure_directory
from pamola_core.utils.ops.op_data_writer import WriterResult, DataWriteError

# Type variable for DataFrames
DataFrameType = TypeVar('DataFrameType', bound=pd.DataFrame)


class MockDataSource:
    """
    A test double for DataSource that works with in-memory DataFrames.

    This class simulates the pamola_core functionality of DataSource without performing
    any actual I/O operations, making it suitable for unit testing operations.
    """

    def __init__(self, dataframes: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Initialize a MockDataSource with optional in-memory DataFrames.

        Parameters:
        -----------
        dataframes : Dict[str, pd.DataFrame], optional
            Dictionary mapping dataset names to pandas DataFrames
        """
        self.dataframes = dataframes or {}
        self.logger = custom_logging.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug("Initializing MockDataSource")

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, name: str = "main") -> 'MockDataSource':
        """
        Create a MockDataSource from a single DataFrame.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to use
        name : str
            Name for the DataFrame (default: "main")

        Returns:
        --------
        MockDataSource
            Initialized MockDataSource with the DataFrame
        """
        return cls({name: df})

    def add_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Add a DataFrame to the MockDataSource.

        Parameters:
        -----------
        name : str
            Name for the DataFrame
        df : pd.DataFrame
            The DataFrame to add
        """
        self.dataframes[name] = df
        self.logger.debug(f"Added DataFrame '{name}' with {len(df)} rows")

    def get_dataframe(self, name: str, **kwargs) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """
        Get a DataFrame by name.

        Parameters:
        -----------
        name : str
            Name of the DataFrame to retrieve
        **kwargs : dict
            Additional parameters (ignored in the mock)

        Returns:
        --------
        Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]
            Tuple containing (DataFrame or None, error_info or None)
        """
        if name in self.dataframes:
            return self.dataframes[name].copy(), None
        else:
            error_info = {
                "error_type": "KeyError",
                "message": f"DataFrame '{name}' not found in MockDataSource",
                "resolution": "Check the dataset name or add it to the MockDataSource"
            }
            self.logger.error(error_info["message"])
            return None, error_info

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get basic schema information for a DataFrame.

        Parameters:
        -----------
        name : str
            Name of the DataFrame

        Returns:
        --------
        Dict[str, Any] or None
            Dictionary with schema information or None if not found
        """
        if name not in self.dataframes:
            return None

        df = self.dataframes[name]

        # Create simple schema
        schema = {
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns},
            "row_count": len(df)
        }

        return schema

    def has_dataframe(self, name: str) -> bool:
        """
        Check if a DataFrame exists by name.

        Parameters:
        -----------
        name : str
            Name to check

        Returns:
        --------
        bool
            True if the DataFrame exists, False otherwise
        """
        return name in self.dataframes

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        self.dataframes.clear()
        return False  # Don't suppress exceptions


class CallRecord(NamedTuple):
    """Record of a method call with parameters and result."""
    method: str
    params: Dict[str, Any]
    result: Any
    timestamp: datetime


class StubDataWriter:
    """
    A test double for DataWriter that writes to a temporary directory.

    This class captures all calls to writing methods and performs real writes
    to a temporary directory for verification in tests.
    """

    def __init__(self, task_dir: Optional[Path] = None, logger=None):
        """
        Initialize a StubDataWriter.

        Parameters:
        -----------
        task_dir : Path, optional
            Base directory for task outputs and artifacts.
            If None, a temporary directory is created.
        logger : logging.Logger, optional
            Logger instance for output messages
        """
        # Create a temporary directory if task_dir is not provided
        self._temp_dir = None
        if task_dir is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            self.task_dir = Path(self._temp_dir.name)
        else:
            self.task_dir = Path(task_dir) if isinstance(task_dir, str) else task_dir

        # Initialize logger
        self.logger = logger or custom_logging.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initializing StubDataWriter for task_dir: {self.task_dir}")

        # Record of all calls made to this stub
        self.calls: List[CallRecord] = []

        # Create pamola_core directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create the standard directory structure if it doesn't exist."""
        # Create main task directory
        ensure_directory(self.task_dir)

        # Create standard subdirectories
        ensure_directory(self.task_dir / "output")
        ensure_directory(self.task_dir / "dictionaries")
        ensure_directory(self.task_dir / "logs")

        self.logger.debug(f"Initialized directory structure under {self.task_dir}")

    def _record_call(self, method: str, params: Dict[str, Any], result: Any) -> None:
        """
        Record a method call for later inspection.

        Parameters:
        -----------
        method : str
            Name of the method called
        params : Dict[str, Any]
            Parameters passed to the method
        result : Any
            Result returned by the method
        """
        call = CallRecord(
            method=method,
            params=params,
            result=result,
            timestamp=datetime.now()
        )
        self.calls.append(call)

    def _get_output_path(self,
                         name: str,
                         extension: str,
                         subdir: Optional[str] = None,
                         timestamp_in_name: bool = False) -> Path:
        """
        Generate the complete output path for a file.

        Parameters:
        -----------
        name : str
            Base name for the output file (without extension)
        extension : str
            File extension (without leading dot)
        subdir : str, optional
            Subdirectory under task_dir, if any
        timestamp_in_name : bool
            Whether to include a timestamp in the filename

        Returns:
        --------
        Path
            Complete path for the output file
        """
        # Ensure extension starts with a dot
        if not extension.startswith('.'):
            extension = f".{extension}"

        # Add timestamp if requested
        if timestamp_in_name:
            timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
            filename = f"{timestamp}_{name}{extension}"
        else:
            filename = f"{name}{extension}"

        # Determine base directory
        if subdir:
            # Create subdir if it doesn't exist
            base_dir = self.task_dir / subdir
            ensure_directory(base_dir)
        else:
            base_dir = self.task_dir

        return base_dir / filename

    def write_dataframe(self,
                        df: pd.DataFrame,
                        name: str,
                        format: str = "csv",
                        subdir: str = "output",
                        timestamp_in_name: bool = False,
                        encryption_key: Optional[str] = None,
                        overwrite: bool = True,
                        **kwargs) -> WriterResult:
        """
        Write a DataFrame to a file within the task directory structure.

        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to be written
        name : str
            Base name for the output file (without extension)
        format : str
            Output format - "csv", "parquet", etc.
        subdir : str
            Subdirectory under task_dir, default "output"
        timestamp_in_name : bool
            Whether to include a timestamp in the filename
        encryption_key : str, optional
            Key for encrypting the output file (ignored in stub)
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the specific writer function

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        # Get the output path
        output_path = self._get_output_path(
            name,
            extension=format,
            subdir=subdir,
            timestamp_in_name=timestamp_in_name
        )

        # Check if file exists and we're not overwriting
        if output_path.exists() and not overwrite:
            raise DataWriteError(f"File {output_path} already exists and overwrite=False")

        # Log writing operation
        self.logger.info(f"[STUB] Writing {format.upper()} data to {output_path}")

        try:
            # Actually write the file (for verification purposes)
            if format.lower() == "csv":
                df.to_csv(output_path, **kwargs)
            elif format.lower() in ("parquet", "pq"):
                # Check if pyarrow is available
                try:
                    import pyarrow
                    df.to_parquet(output_path, **kwargs)
                except ImportError:
                    # Fallback to CSV if pyarrow not available
                    self.logger.warning("pyarrow not available, falling back to CSV")
                    output_path = output_path.with_suffix(".csv")
                    df.to_csv(output_path, **kwargs)
            else:
                raise DataWriteError(f"Unsupported format in stub: {format}")

            # Get file stats
            file_size = output_path.stat().st_size
            timestamp = datetime.fromtimestamp(output_path.stat().st_mtime)

            result = WriterResult(
                path=output_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format=format.lower()
            )

            # Record the call
            self._record_call(
                method="write_dataframe",
                params={
                    "name": name,
                    "format": format,
                    "subdir": subdir,
                    "timestamp_in_name": timestamp_in_name,
                    "has_encryption_key": encryption_key is not None,
                    "overwrite": overwrite,
                    "kwargs": kwargs,
                    "df_shape": df.shape,
                    "df_columns": list(df.columns)
                },
                result=result
            )

            return result

        except Exception as e:
            # Log error
            self.logger.error(f"[STUB] Error writing DataFrame: {str(e)}")

            # Record the failed call
            self._record_call(
                method="write_dataframe",
                params={
                    "name": name,
                    "format": format,
                    "subdir": subdir,
                    "timestamp_in_name": timestamp_in_name,
                    "has_encryption_key": encryption_key is not None,
                    "overwrite": overwrite,
                    "df_shape": df.shape,
                    "error": str(e)
                },
                result=None
            )

            # Re-raise
            raise DataWriteError(f"[STUB] Failed to write DataFrame: {str(e)}") from e

    def write_json(self,
                   data: Dict[str, Any],
                   name: str,
                   subdir: Optional[str] = None,
                   timestamp_in_name: bool = False,
                   encryption_key: Optional[str] = None,
                   pretty: bool = True,
                   overwrite: bool = True,
                   **kwargs) -> WriterResult:
        """
        Write a JSON object to a file within the task directory structure.

        Parameters:
        -----------
        data : Dict[str, Any]
            JSON-serializable data to write
        name : str
            Base name for the output file (without extension)
        subdir : str, optional
            Subdirectory under task_dir
        timestamp_in_name : bool
            Whether to include a timestamp in the filename
        encryption_key : str, optional
            Key for encrypting the output file (ignored in stub)
        pretty : bool
            Whether to pretty-print the JSON with indentation
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the json writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        # Get the output path
        output_path = self._get_output_path(
            name,
            extension="json",
            subdir=subdir,
            timestamp_in_name=timestamp_in_name
        )

        # Check if file exists and we're not overwriting
        if output_path.exists() and not overwrite:
            raise DataWriteError(f"File {output_path} already exists and overwrite=False")

        # Log writing operation
        self.logger.info(f"[STUB] Writing JSON data to {output_path}")

        try:
            # Set indent if pretty-printing
            indent = 2 if pretty else None

            # Write JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, **kwargs) # type: ignore

            # Get file stats
            file_size = output_path.stat().st_size
            timestamp = datetime.fromtimestamp(output_path.stat().st_mtime)

            result = WriterResult(
                path=output_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format="json"
            )

            # Record the call
            self._record_call(
                method="write_json",
                params={
                    "name": name,
                    "subdir": subdir,
                    "timestamp_in_name": timestamp_in_name,
                    "has_encryption_key": encryption_key is not None,
                    "pretty": pretty,
                    "overwrite": overwrite,
                    "kwargs": kwargs,
                    "data_keys": list(data.keys() if isinstance(data, dict) else [])
                },
                result=result
            )

            return result

        except Exception as e:
            # Log error
            self.logger.error(f"[STUB] Error writing JSON: {str(e)}")

            # Record the failed call
            self._record_call(
                method="write_json",
                params={
                    "name": name,
                    "subdir": subdir,
                    "timestamp_in_name": timestamp_in_name,
                    "has_encryption_key": encryption_key is not None,
                    "pretty": pretty,
                    "overwrite": overwrite,
                    "error": str(e)
                },
                result=None
            )

            # Re-raise
            raise DataWriteError(f"[STUB] Failed to write JSON: {str(e)}") from e

    def write_metrics(self,
                      metrics: Dict[str, Any],
                      name: str,
                      timestamp_in_name: bool = True,
                      encryption_key: Optional[str] = None,
                      overwrite: bool = True,
                      **kwargs) -> WriterResult:
        """
        Save metrics to the root task directory.

        Parameters:
        -----------
        metrics : Dict[str, Any]
            Metrics data to save
        name : str
            Base name for the output file (without extension)
        timestamp_in_name : bool
            Whether to include a timestamp in the filename (default True)
        encryption_key : str, optional
            Key for encrypting the output file (ignored in stub)
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        # Add metadata to metrics
        enriched_metrics = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "name": name,
                "stub": True
            },
            "metrics": metrics
        }

        # Write to the root directory
        return self.write_json(
            enriched_metrics,
            name,
            subdir=None,  # Root directory
            timestamp_in_name=timestamp_in_name,
            encryption_key=encryption_key,
            overwrite=overwrite,
            pretty=True,
            **kwargs
        )

    def write_dictionary(self,
                         data: Dict[str, Any],
                         name: str,
                         timestamp_in_name: bool = False,
                         encryption_key: Optional[str] = None,
                         overwrite: bool = True,
                         format: str = "json",
                         **kwargs) -> WriterResult:
        """
        Save a dictionary to the dictionaries subdirectory.

        Parameters:
        -----------
        data : Dict[str, Any]
            Dictionary to save
        name : str
            Base name for the output file (without extension)
        timestamp_in_name : bool
            Whether to include a timestamp in the filename
        encryption_key : str, optional
            Key for encrypting the output file (ignored in stub)
        overwrite : bool
            Whether to overwrite an existing file
        format : str
            Output format (json, csv, parquet)
        **kwargs
            Additional arguments for the writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata
        """
        # Choose the appropriate method based on format
        if format.lower() == "json":
            return self.write_json(
                data,
                name,
                subdir="dictionaries",
                timestamp_in_name=timestamp_in_name,
                encryption_key=encryption_key,
                overwrite=overwrite,
                **kwargs
            )
        elif format.lower() in ("csv", "parquet", "pq"):
            # Convert dictionary to DataFrame
            if isinstance(data, dict):
                # Check structure
                if all(isinstance(v, (list, tuple)) for v in data.values()):
                    # Dict of lists -> DataFrame columns
                    df = pd.DataFrame(data)
                elif all(isinstance(v, dict) for v in data.values()):
                    # Dict of dicts -> DataFrame with index
                    df = pd.DataFrame.from_dict(data, orient='index')
                else:
                    # Simple dict -> single row DataFrame
                    df = pd.DataFrame([data])
            else:
                raise DataWriteError(f"Cannot convert {type(data).__name__} to DataFrame")

            # Write the DataFrame
            return self.write_dataframe(
                df,
                name,
                format=format,
                subdir="dictionaries",
                timestamp_in_name=timestamp_in_name,
                encryption_key=encryption_key,
                overwrite=overwrite,
                **kwargs
            )
        else:
            raise DataWriteError(f"Unsupported format for dictionaries: {format}")

    def get_output_dir(self) -> Path:
        """
        Get the output directory path.

        Returns:
        --------
        Path
            Path to the output directory
        """
        return self.task_dir / "output"

    def get_dictionaries_dir(self) -> Path:
        """
        Get the dictionaries directory path.

        Returns:
        --------
        Path
            Path to the dictionaries directory
        """
        return self.task_dir / "dictionaries"

    def get_temp_dir(self) -> Optional[tempfile.TemporaryDirectory]:
        """
        Get the temporary directory (if created).

        Returns:
        --------
        tempfile.TemporaryDirectory or None
            Temporary directory if one was created
        """
        return self._temp_dir

    def get_calls(self, method_name: Optional[str] = None) -> List[CallRecord]:
        """
        Get recorded calls, optionally filtered by method name.

        Parameters:
        -----------
        method_name : str, optional
            Method name to filter by

        Returns:
        --------
        List[CallRecord]
            List of matching call records
        """
        if method_name:
            return [call for call in self.calls if call.method == method_name]
        return self.calls

    def clear_calls(self) -> None:
        """Clear the recorded calls."""
        self.calls.clear()

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context."""
        if self._temp_dir:
            self._temp_dir.cleanup()
        return False  # Don't suppress exceptions


def assert_artifact_exists(task_dir: Path, subdir: str, filename_pattern: str) -> Path:
    """
    Assert that a file matching the given pattern exists in the specified directory.

    Parameters:
    -----------
    task_dir : Path
        Base task directory
    subdir : str
        Subdirectory under task_dir
    filename_pattern : str
        Regex pattern to match filenames

    Returns:
    --------
    Path
        Path to the matched file

    Raises:
    -------
    AssertionError
        If no matching file is found
    """
    # Construct the directory path
    dir_path = task_dir / subdir if subdir else task_dir

    # Compile the regex pattern
    pattern = re.compile(filename_pattern)

    # Check if directory exists
    if not dir_path.exists() or not dir_path.is_dir():
        raise AssertionError(f"Directory does not exist: {dir_path}")

    # Look for matching files
    matches = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and pattern.search(file_path.name):
            matches.append(file_path)

    # Assert that we found at least one match
    if not matches:
        raise AssertionError(
            f"No file matching pattern '{filename_pattern}' found in {dir_path}.\n"
            f"Directory contains: {[f.name for f in dir_path.iterdir() if f.is_file()]}"
        )

    # If multiple matches, return the most recent one
    if len(matches) > 1:
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return matches[0]


def assert_metrics_content(task_dir: Path, expected_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assert that metrics file contains the expected metrics.

    This performs a partial match, ensuring all keys in expected_metrics
    exist and match, while allowing extra fields.

    Parameters:
    -----------
    task_dir : Path
        Base task directory
    expected_metrics : Dict[str, Any]
        Dictionary with expected metrics

    Returns:
    --------
    Dict[str, Any]
        The full metrics content

    Raises:
    -------
    AssertionError
        If metrics file is not found or content doesn't match expectations
    """
    # Look for metrics files in the root directory
    metrics_files = list(task_dir.glob("*.json"))

    # Filter for ones that might be metrics
    metrics_files = [f for f in metrics_files if "metrics" in f.name.lower()]

    if not metrics_files:
        raise AssertionError(f"No metrics file found in {task_dir}")

    # Sort by modification time to get the most recent
    metrics_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    metrics_file = metrics_files[0]

    # Load the metrics file
    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_content = json.load(f)
    except Exception as e:
        raise AssertionError(f"Error loading metrics file {metrics_file}: {str(e)}")

    # If metrics are nested under a 'metrics' key, use that
    if "metrics" in metrics_content and isinstance(metrics_content["metrics"], dict):
        actual_metrics = metrics_content["metrics"]
    else:
        actual_metrics = metrics_content

    # Perform a partial match against expected metrics
    missing_keys = []
    mismatched_values = []

    def _compare_nested_dict(expected, actual, path=""):
        for key, expected_value in expected.items():
            current_path = f"{path}.{key}" if path else key

            if key not in actual:
                missing_keys.append(current_path)
                continue

            actual_value = actual[key]

            # Recursive comparison for nested dictionaries
            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                _compare_nested_dict(expected_value, actual_value, current_path)
            elif expected_value != actual_value:
                mismatched_values.append((current_path, expected_value, actual_value))

    # Compare the dictionaries
    _compare_nested_dict(expected_metrics, actual_metrics)

    # Build assertion message if there are issues
    if missing_keys or mismatched_values:
        message = "Metrics content did not match expectations:\n"

        if missing_keys:
            message += f"Missing keys: {', '.join(missing_keys)}\n"

        if mismatched_values:
            message += "Mismatched values:\n"
            for path, expected, actual in mismatched_values:
                message += f"  {path}: expected {expected}, got {actual}\n"

        raise AssertionError(message)

    return metrics_content


def create_test_operation_env(tmp_path: Path, config_overrides: Optional[Dict[str, Any]] = None) -> Tuple[Path, Any]:
    """
    Create a test environment for operations with a temporary task directory
    and a minimal OperationConfig.

    Parameters:
    -----------
    tmp_path : Path
        Base path for creating the test environment
    config_overrides : Dict[str, Any], optional
        Overrides for the default configuration

    Returns:
    --------
    Tuple[Path, Any]
        Tuple containing (task_dir, operation_config)
    """
    # Import here to avoid circular imports
    try:
        from pamola_core.utils.ops.op_config import OperationConfig
    except ImportError:
        # Mock OperationConfig if not available
        class MockOperationConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

            def to_dict(self):
                return {key: getattr(self, key) for key in dir(self)
                        if not key.startswith('_') and not callable(getattr(self, key))}

        OperationConfig = MockOperationConfig

    # Create task directory
    task_dir = tmp_path / "task"
    ensure_directory(task_dir)

    # Create standard subdirectories
    ensure_directory(task_dir / "output")
    ensure_directory(task_dir / "dictionaries")
    ensure_directory(task_dir / "logs")

    # Create default configuration
    default_config = {
        "operation_name": "test_operation",
        "version": "1.0.0",
        "description": "Test operation for unit testing",
        "parameters": {
            "field_name": "test_field",
            "threshold": 0.5
        }
    }

    # Apply overrides if provided
    if config_overrides:
        # Handle nested dictionaries correctly
        def deep_update(source, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and key in source and isinstance(source[key], dict):
                    deep_update(source[key], value)
                else:
                    source[key] = value

        deep_update(default_config, config_overrides)

    # Write config.json
    config_path = task_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, indent=2) # type: ignore

    # Create OperationConfig instance
    op_config = OperationConfig(**default_config)

    return task_dir, op_config