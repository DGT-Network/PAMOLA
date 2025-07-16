"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Data Writer
Description: Unified data writing interface for operations
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides a specialized writer for operation outputs,
ensuring consistent handling of files, paths, encryption, and
directory structure across all PAMOLA Core operations.

Key features:
- Unified writing interface for multiple data formats
- Structured artifact organization under task_dir
- Transparent encryption support
- Automatic directory creation
- Integration with progress tracking
- Dask DataFrame support for large datasets
"""

import inspect
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Union, Optional, TypeVar, NamedTuple

import dask.dataframe as dd
import pandas as pd

from pamola_core.utils import logging as custom_logging
from pamola_core.utils.io import (
    # Pamola Core writing functions
    write_dataframe_to_csv, write_json, write_parquet,
    save_visualization, save_plot, append_to_json_array,
    merge_json_objects, ensure_directory,
    # Helpers
    get_timestamped_filename
)
from pamola_core.utils.progress import HierarchicalProgressTracker


# Define type variables for better type hints
PathType = Union[str, Path]
DataFrameType = TypeVar('DataFrameType', bound=pd.DataFrame)
DaskDataFrameType = TypeVar('DaskDataFrameType', bound=dd.DataFrame)


class WriterResult(NamedTuple):
    """Result of a write operation, including metadata."""
    path: Path
    size_bytes: int
    timestamp: datetime
    format: str


class DataWriteError(Exception):
    """Exception raised for errors during data writing operations."""
    pass


class DataWriter:
    """
    Specialized writer for operation outputs with structured organization.

    This class provides a consistent interface for writing operation outputs
    to the appropriate locations within the task directory structure, with
    optional encryption, progress tracking, and special handling for large datasets.
    """

    def __init__(
            self,
            *,  # Force keyword-only arguments for clarity
            task_dir: Union[str, Path],
            logger: Optional[logging.Logger] = None,
            progress_tracker: Optional[HierarchicalProgressTracker] = None,
            use_encryption: bool = False,
            encryption_key: Optional[bytes] = None,
            encryption_mode: str = "none"
    ):
        """
        Initialize a DataWriter instance with the specified task directory.

        Parameters:
        -----------
        task_dir : Union[str, Path]
            Base directory for task outputs and artifacts
        logger : logging.Logger, optional
            Logger instance for output messages (created if not provided)
        progress_tracker : HierarchicalProgressTracker, optional
            Progress tracker for reporting writing progress
        use_encryption : bool, optional
            Whether to use encryption for sensitive outputs
        encryption_key : bytes, optional
            Encryption key to use if encryption is enabled
        encryption_mode : str, optional
            Encryption mode to use ("none", "simple", "age")
        """
        # Convert task_dir to Path if it's a string
        self.task_dir = Path(task_dir) if isinstance(task_dir, str) else task_dir

        # Initialize logger
        self.logger = logger or custom_logging.get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initializing DataWriter for task_dir: {self.task_dir}")

        # Set progress tracker
        self.progress_tracker = progress_tracker

        # Set encryption parameters
        self.use_encryption = use_encryption
        self.encryption_key = encryption_key
        self.encryption_mode = encryption_mode

        if use_encryption:
            self.logger.debug(f"Encryption enabled with mode: {encryption_mode}")
            if encryption_key is None:
                self.logger.warning("Encryption enabled but no key provided")

        # Create pamola core directories
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """
        Create the standard directory structure if it doesn't exist.
        """
        # Create main task directory
        ensure_directory(self.task_dir)

        # Create standard subdirectories
        ensure_directory(self.task_dir / "output")
        ensure_directory(self.task_dir / "dictionaries")
        ensure_directory(self.task_dir / "logs")
        ensure_directory(self.task_dir / "visualizations")

        self.logger.debug(f"Initialized directory structure under {self.task_dir}")

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
        # Ensure extension not starts with a dot
        if extension.startswith('.'):
            extension = extension.lstrip('.')

        # Add timestamp if requested
        if timestamp_in_name:
            filename = get_timestamped_filename(name, extension)
        else:
            filename = f"{name}.{extension}"

        # Determine base directory
        if subdir:
            # Create subdir if it doesn't exist
            base_dir = self.task_dir / subdir
            ensure_directory(base_dir)
        else:
            base_dir = self.task_dir

        return base_dir / filename

    def write_dataframe(self,
                        df: Union[pd.DataFrame, dd.DataFrame],
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
        df : Union[pd.DataFrame, dd.DataFrame]
            DataFrame or Dask DataFrame to be written
        name : str
            Base name for the output file (without extension)
        format : str
            Output format - "csv", "parquet", etc.
        subdir : str
            Subdirectory under task_dir, default "output"
        timestamp_in_name : bool
            Whether to include a timestamp in the filename
        encryption_key : str, optional
            Key for encrypting the output file
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the specific writer function

        Returns:
        --------
        WriterResult
            Result object with path and metadata

        Raises:
        -------
        DataWriteError
            If the write operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        # Create a progress subtask if we have a tracker
        subtask = None
        if self.progress_tracker:
            # Configure description based on what we're writing
            action = "Encrypting and writing" if encryption_key else "Writing"
            description = f"{action} {format.upper()} output: {name}"

            # Create a subtask with an estimate of work
            # For Dask, we need to materialize to get row count, so use a reasonable default
            if isinstance(df, dd.DataFrame):
                total = 100  # Use a reasonable default for Dask
            else:
                total = min(len(df), 100)  # Cap at 100 for regular DataFrames

            subtask = self.progress_tracker.create_subtask(
                total=total,
                description=description,
                unit="steps"
            )

        try:
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
            self.logger.info(f"Writing {format.upper()} data to {Path(output_path).name}")

            # Handle Dask DataFrames specially
            if isinstance(df, dd.DataFrame):
                return self._write_dask_dataframe(
                    df, output_path, format, encryption_key, subtask, **kwargs
                )

            # For regular DataFrames, use the appropriate io function
            if format.lower() == "csv":
                written_path = write_dataframe_to_csv(
                    df,
                    output_path,
                    encryption_key=encryption_key,
                    **kwargs
                )
            elif format.lower() in ("parquet", "pq"):
                written_path = write_parquet(
                    df,
                    output_path,
                    encryption_key=encryption_key,
                    **kwargs
                )
            elif format.lower() == "json":
                written_path = write_json(
                    df.to_dict(orient="records"),
                    output_path,
                    encryption_key=encryption_key,
                    **kwargs
                )
            else:
                raise DataWriteError(f"Unsupported format: {format}")

            # Update progress
            if subtask:
                subtask.update(subtask.total)

            # Get file stats
            file_size = written_path.stat().st_size
            timestamp = datetime.fromtimestamp(written_path.stat().st_mtime)

            return WriterResult(
                path=written_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format=format.lower()
            )

        except Exception as e:
            # Log error and re-raise as DataWriteError
            self.logger.error(f"Error writing DataFrame: {str(e)}")

            # Update progress with error
            if subtask:
                subtask.update(0, {"status": "error", "message": str(e)})

            # Include original exception details
            raise DataWriteError(f"Failed to write DataFrame: {str(e)}") from e

        finally:
            # Close subtask
            if subtask:
                subtask.close()

    def _write_dask_dataframe(self,
                              df: dd.DataFrame,
                              output_path: Path,
                              format: str,
                              encryption_key: Optional[str],
                              subtask: Optional[HierarchicalProgressTracker] = None,
                              **kwargs) -> WriterResult:
        """
        Write a Dask DataFrame, handling partitioning appropriately.

        Parameters:
        -----------
        df : dd.DataFrame
            Dask DataFrame to write
        output_path : Path
            Target output path
        format : str
            Output format
        encryption_key : str, optional
            Encryption key
        subtask : HierarchicalProgressTracker, optional
            Progress tracker for this operation
        **kwargs
            Additional arguments for the writer

        Returns:
        --------
        WriterResult
            Result with path and metadata
        """
        # For Dask DataFrames, we need different approaches based on format
        try:
            # For parquet, we can write to a directory
            if format.lower() in ("parquet", "pq"):
                # For Dask, we may need to write to a directory instead of a single file
                output_dir = output_path.parent / output_path.stem
                ensure_directory(output_dir)

                # Write Dask DataFrame to parquet
                df.to_parquet(output_dir, **kwargs)

                # Log success
                self.logger.info(f"Dask DataFrame written to {output_dir} (partitioned parquet)")

                # If encryption is needed, this is more complex - we'd need to encrypt each file
                if encryption_key:
                    self.logger.warning(
                        "Encryption for partitioned Dask output is not fully implemented. "
                        "Consider materializing the DataFrame for secure encryption."
                    )

                # Update progress
                if subtask:
                    subtask.update(subtask.total)

                # For a directory, we return the directory path
                return WriterResult(
                    path=output_dir,
                    size_bytes=sum(f.stat().st_size for f in output_dir.glob("*.parquet")),
                    timestamp=datetime.now(),
                    format=f"partitioned_{format.lower()}"
                )

            # For CSV, we can write to a single file if it's not too large
            elif format.lower() == "csv":
                # Get partition count - if manageable, compute and write as regular DF
                if df.npartitions <= 10:  # Arbitrary threshold
                    self.logger.info(f"Converting Dask DataFrame ({df.npartitions} partitions) to pandas")

                    # Compute the DataFrame
                    regular_df = df.compute()

                    # Write using the regular method
                    return self.write_dataframe(
                        regular_df,
                        name=output_path.stem,
                        format=format,
                        subdir="",  # Skip subdir as we're using an absolute path
                        timestamp_in_name=False,  # Skip timestamp as it's in the path
                        encryption_key=encryption_key,
                        overwrite=True,
                        **kwargs
                    )
                else:
                    # For many partitions, write partitioned CSVs
                    output_dir = output_path.parent / output_path.stem
                    ensure_directory(output_dir)

                    # Write partitioned CSVs
                    df.to_csv(output_dir / "part-*.csv", **kwargs)

                    # Log success
                    self.logger.info(f"Large Dask DataFrame written to {output_dir} (partitioned CSV)")

                    # Warning for encryption
                    if encryption_key:
                        self.logger.warning(
                            "Encryption for partitioned Dask output is not fully implemented. "
                            "Consider materializing the DataFrame for secure encryption."
                        )

                    # Update progress
                    if subtask:
                        subtask.update(subtask.total)

                    # Return directory result
                    return WriterResult(
                        path=output_dir,
                        size_bytes=sum(f.stat().st_size for f in output_dir.glob("*.csv")),
                        timestamp=datetime.now(),
                        format=f"partitioned_{format.lower()}"
                    )
            else:
                raise DataWriteError(f"Unsupported format for Dask DataFrames: {format}")

        except Exception as e:
            # Log error
            self.logger.error(f"Error writing Dask DataFrame: {str(e)}")

            # Update progress
            if subtask:
                subtask.update(0, {"status": "error", "message": str(e)})

            # Re-raise
            raise DataWriteError(f"Failed to write Dask DataFrame: {str(e)}") from e

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
            Key for encrypting the output file
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

        Raises:
        -------
        DataWriteError
            If the write operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        try:
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
            self.logger.info(f"Writing JSON data to {Path(output_path).name}")

            # Set indent if pretty-printing
            if "indent" not in kwargs and pretty:
                kwargs["indent"] = 2

            # Write JSON using io module
            written_path = write_json(
                data,
                output_path,
                encryption_key=encryption_key,
                **kwargs
            )

            # Get file stats
            file_size = written_path.stat().st_size
            timestamp = datetime.fromtimestamp(written_path.stat().st_mtime)

            return WriterResult(
                path=written_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format="json"
            )

        except Exception as e:
            # Log error
            self.logger.error(f"Error writing JSON: {str(e)}")

            # Re-raise as DataWriteError
            raise DataWriteError(f"Failed to write JSON: {str(e)}") from e

    def append_to_json_array(self,
                             item: Any,
                             name: str,
                             subdir: Optional[str] = None,
                             encryption_key: Optional[str] = None,
                             create_if_missing: bool = True,
                             pretty: bool = True,
                             **kwargs) -> WriterResult:
        """
        Append an item to a JSON array file.

        Parameters:
        -----------
        item : Any
            JSON-serializable item to append
        name : str
            Base name for the JSON array file (without extension)
        subdir : str, optional
            Subdirectory under task_dir
        encryption_key : str, optional
            Key for encrypting the output file
        create_if_missing : bool
            Whether to create the file if it doesn't exist
        pretty : bool
            Whether to pretty-print the JSON with indentation
        **kwargs
            Additional arguments for the json writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata

        Raises:
        -------
        DataWriteError
            If the append operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        try:
            # Get the output path
            output_path = self._get_output_path(
                name,
                extension="json",
                subdir=subdir,
                timestamp_in_name=False  # Don't timestamp append files
            )

            # Log operation
            self.logger.info(f"Appending to JSON array at {output_path}")

            # Set indent if pretty-printing
            if "indent" not in kwargs and pretty:
                kwargs["indent"] = 2

            # Append using io module
            written_path = append_to_json_array(
                item,
                output_path,
                encryption_key=encryption_key,
                create_if_missing=create_if_missing,
                **kwargs
            )

            # Get file stats
            file_size = written_path.stat().st_size
            timestamp = datetime.fromtimestamp(written_path.stat().st_mtime)

            return WriterResult(
                path=written_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format="json"
            )

        except Exception as e:
            # Log error
            self.logger.error(f"Error appending to JSON array: {str(e)}")

            # Re-raise
            raise DataWriteError(f"Failed to append to JSON array: {str(e)}") from e

    def merge_json_objects(self,
                           data: Dict[str, Any],
                           name: str,
                           subdir: Optional[str] = None,
                           encryption_key: Optional[str] = None,
                           create_if_missing: bool = True,
                           overwrite_existing: bool = True,
                           recursive_merge: bool = True,
                           pretty: bool = True,
                           **kwargs) -> WriterResult:
        """
        Merge data with an existing JSON object file.

        Parameters:
        -----------
        data : Dict[str, Any]
            JSON-serializable data to merge
        name : str
            Base name for the JSON file (without extension)
        subdir : str, optional
            Subdirectory under task_dir
        encryption_key : str, optional
            Key for encrypting the output file
        create_if_missing : bool
            Whether to create the file if it doesn't exist
        overwrite_existing : bool
            Whether to overwrite existing keys
        recursive_merge : bool
            Whether to merge nested dictionaries recursively
        pretty : bool
            Whether to pretty-print the JSON with indentation
        **kwargs
            Additional arguments for the json writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata

        Raises:
        -------
        DataWriteError
            If the merge operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        try:
            # Get the output path
            output_path = self._get_output_path(
                name,
                extension="json",
                subdir=subdir,
                timestamp_in_name=False  # Don't timestamp merge files
            )

            # Log operation
            self.logger.info(f"Merging JSON objects at {output_path}")

            # Set indent if pretty-printing
            if "indent" not in kwargs and pretty:
                kwargs["indent"] = 2

            # Merge using io module
            written_path = merge_json_objects(
                data,
                output_path,
                encryption_key=encryption_key,
                create_if_missing=create_if_missing,
                overwrite_existing=overwrite_existing,
                recursive_merge=recursive_merge,
                **kwargs
            )

            # Get file stats
            file_size = written_path.stat().st_size
            timestamp = datetime.fromtimestamp(written_path.stat().st_mtime)

            return WriterResult(
                path=written_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format="json"
            )

        except Exception as e:
            # Log error
            self.logger.error(f"Error merging JSON objects: {str(e)}")

            # Re-raise
            raise DataWriteError(f"Failed to merge JSON objects: {str(e)}") from e

    def write_visualization(self,
                            figure: Any,
                            name: str,
                            subdir: Optional[str] = None,
                            timestamp_in_name: bool = False,
                            format: str = "png",
                            encryption_key: Optional[str] = None,
                            overwrite: bool = True,
                            **kwargs) -> WriterResult:
        """
        Save a visualization figure to a file.

        Parameters:
        -----------
        figure : Any
            Visualization figure (matplotlib, plotly, etc.)
        name : str
            Base name for the output file (without extension)
        subdir : str, optional
            Subdirectory under task_dir
        timestamp_in_name : bool
            Whether to include a timestamp in the filename
        format : str
            Output format (png, jpg, svg, etc.)
        encryption_key : str, optional
            Key for encrypting the output file
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the visualization saver

        Returns:
        --------
        WriterResult
            Result object with path and metadata

        Raises:
        -------
        DataWriteError
            If the save operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        try:
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
            self.logger.info(f"Saving visualization to {output_path}")

            # Use appropriate function based on figure type
            if "matplotlib" in str(type(figure).__module__):
                # For matplotlib
                written_path = save_plot(
                    figure,
                    output_path,
                    encryption_key=encryption_key,
                    **kwargs
                )
            else:
                # For other visualization libraries
                written_path = save_visualization(
                    figure,
                    output_path,
                    format=format,
                    encryption_key=encryption_key,
                    **kwargs
                )

            # Get file stats
            file_size = written_path.stat().st_size
            timestamp = datetime.fromtimestamp(written_path.stat().st_mtime)

            return WriterResult(
                path=written_path,
                size_bytes=file_size,
                timestamp=timestamp,
                format=format.lower()
            )

        except Exception as e:
            # Log error
            self.logger.error(f"Error saving visualization: {str(e)}")

            # Re-raise
            raise DataWriteError(f"Failed to save visualization: {str(e)}") from e

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
            Key for encrypting the output file
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

        Raises:
        -------
        DataWriteError
            If the save operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

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
            Key for encrypting the output file
        overwrite : bool
            Whether to overwrite an existing file
        **kwargs
            Additional arguments for the writer

        Returns:
        --------
        WriterResult
            Result object with path and metadata

        Raises:
        -------
        DataWriteError
            If the save operation fails
        """
        # Use class encryption key if none provided and encryption is enabled
        if encryption_key is None and self.use_encryption:
            encryption_key = self.encryption_key

        # Add metadata to metrics
        enriched_metrics = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "name": name,
                "operation": self._get_caller_info()
            },
            "metrics": metrics
        }

        # Write to the root directory
        return self.write_json(
            enriched_metrics,
            name,
            subdir="metrics",  # Root directory
            timestamp_in_name=timestamp_in_name,
            encryption_key=encryption_key,
            overwrite=overwrite,
            pretty=True,
            **kwargs
        )

    def _get_caller_info(self) -> Dict[str, str]:
        """
        Get information about the calling operation/class.

        Returns:
        --------
        Dict[str, str]
            Information about the caller
        """
        caller_info = {}

        # Try to get more info about the caller
        stack = inspect.stack()
        try:
            # Skip DataWriter methods and look for the operation frame
            for frame_info in stack[1:]:
                frame = frame_info.frame
                module = inspect.getmodule(frame)

                # Skip if no module or if it's this module
                if not module or module.__name__ == __name__:
                    continue

                # Get module info
                caller_info["module"] = module.__name__

                # Try to find class info
                if "self" in frame.f_locals:
                    instance = frame.f_locals["self"]
                    caller_info["class"] = instance.__class__.__name__

                    # If it looks like an operation, add info
                    if hasattr(instance, "__version__"):
                        caller_info["version"] = str(instance.__version__)

                # Get function name
                caller_info["function"] = frame_info.function

                # Stop at first non-DataWriter frame with info
                if caller_info:
                    break

        except Exception as e:
            # If anything goes wrong, just log and continue
            self.logger.debug(f"Error getting caller info: {str(e)}")

        return caller_info


# Alias for backwards compatibility
DataWriterResult = WriterResult