"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module:        Progress Tracking Utilities
Package:       pamola_core.utils
Version:       2.0.0+refactor.2025.05.22
Status:        stable
Author:        PAMOLA Core Team
Created:       2025
License:       BSD 3-Clause
Description:
   Robust progress tracking and visualization for large-scale data processing.
   This module provides comprehensive progress tracking capabilities for processing
   very large datasets in the context of privacy-preserving transformations.

Key Features:
   - Real-time progress visualization with ETA and memory usage monitoring
   - Multi-stage process tracking for complex data pipelines
   - Memory-efficient chunked processing for datasets with millions of records
   - Hierarchical progress reporting for nested operations
   - Error handling and recovery mechanisms
   - Parallelization support with coordinated progress tracking
   - Clean and intuitive API with backward compatibility guarantees
   - Proxy attributes for seamless integration with tqdm internals

Framework:
   This module is part of PAMOLA.CORE's utilities and provides progress tracking
   for all I/O and processing operations throughout the framework.

Changelog:
   2.0.0 (2025-05-22): Major refactoring
       - Added proxy properties (n, total, elapsed) to ProgressBase for tqdm attribute access
       - Added setter for total property to support dynamic updates
       - Consolidated SimpleProgressBar and ProgressBar to reduce duplication
       - Improved error handling in context managers
       - Enhanced memory tracking accuracy
       - Better integration with I/O module progress calculations
   1.3.0 (2025-05-01): Added enhanced parallel processing support
   1.2.0 (2025-04-15): Added hierarchical progress tracking
   1.1.0 (2025-03-01): Added memory usage tracking
   1.0.0 (2025-01-01): Initial release

TODO:
   - Add cloud-based distributed progress tracking for cluster operations
   - Implement dynamic rate limiting based on resource utilization
   - Enhance memory profiling with per-object type breakdowns
   - Add progress persistence for resumable operations
   - Integrate with system-wide notification mechanisms
   - Add benchmarking tools and performance comparison metrics
"""

import functools
import logging
import sys
import time
import warnings
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Callable, Iterator

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# ======= Configurable logging system =======
# Default configuration that can be overridden
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FORMAT = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
_DEFAULT_LOG_HANDLERS: Optional[List[logging.Handler]] = (
    None  # Will be initialized on first use
)

_logger: Optional[logging.Logger] = None


def configure_logging(
    level: int = _DEFAULT_LOG_LEVEL,
    format_str: str = _DEFAULT_LOG_FORMAT,
    handlers: Optional[List[logging.Handler]] = None,
    log_file: Optional[str] = "pamola_processing.log",
) -> logging.Logger:
    """
    Configure logging for the progress module.

    Parameters:
    -----------
    level : int
        Logging level (e.g., logging.INFO)
    format_str : str
        Log message format string
    handlers : List[logging.Handler], optional
        Custom log handlers
    log_file : str, optional
        Path to log file, None to disable file logging

    Returns:
    --------
    logging.Logger
        Configured logger instance
    """
    global _logger

    if _logger is not None:
        return _logger

    _logger = logging.getLogger("pamola")
    _logger.setLevel(level)

    # Clear any existing handlers
    for handler in _logger.handlers[:]:
        _logger.removeHandler(handler)

    # Use provided handlers or create default ones
    if handlers:
        for handler in handlers:
            _logger.addHandler(handler)
    else:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_str))
        _logger.addHandler(console_handler)

        # Create file handler if log_file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(format_str))
            _logger.addHandler(file_handler)

    return _logger


# Get or initialize logger
def get_logger() -> logging.Logger:
    """Get the module logger, initializing it if necessary."""
    global _logger
    if _logger is None:
        _logger = configure_logging()
    assert _logger is not None, "Logger initialization failed"
    return _logger


# For backwards compatibility
logger = get_logger()


# Decorator for deprecation warnings
def deprecated(func=None, *, alternative=None):
    """
    Mark functions or classes as deprecated to warn users.

    Parameters:
    -----------
    func : callable
        The function or class to mark as deprecated
    alternative : str, optional
        Name of the alternative function or class to use
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            warning_msg = (
                f"{f.__name__} is deprecated and will be removed in a future version."
            )
            if alternative:
                warning_msg += f" Use {alternative} instead."
            warnings.warn(warning_msg, DeprecationWarning, stacklevel=2)
            return f(*args, **kwargs)

        return wrapper

    if func is None:
        return decorator
    return decorator(func)


# ======= Base class for progress tracking =======
class ProgressBase:
    """
    Base class for progress tracking functionality.
    Provides common interface for all progress trackers.
    """

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        bar_format: Optional[str] = None,
        track_memory: bool = False,
    ):
        """
        Initialize progress tracker base.

        Parameters:
        -----------
        total : int, optional
            Total number of items to process
        description : str
            Description of the current operation
        unit : str
            Unit for the progress bar (e.g., "records", "chunks")
        bar_format : str, optional
            Custom format for the progress bar
        track_memory : bool
            Whether to track memory usage during processing
        """
        self._total = total  # Store as private to avoid conflict with property
        self.description = description
        self.unit = unit
        self.track_memory = track_memory
        self.start_time = time.time()
        self.start_memory = (
            psutil.Process().memory_info().rss / (1024 * 1024) if track_memory else 0
        )
        self.peak_memory = self.start_memory

        # Default format if none specified
        if bar_format is None:
            bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        self.pbar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            file=sys.stdout,
            leave=True,
            bar_format=bar_format,
        )

        get_logger().info(
            f"Started: {description}" + (f" (total: {total} {unit})" if total else "")
        )

    # --- Proxy attributes to underlying tqdm object ---
    @property
    def n(self) -> int:
        """Current counter value (proxy to tqdm.n)."""
        return self.pbar.n if self.pbar else 0

    @property
    def total(self) -> Optional[int]:
        """Total expected iterations (proxy to tqdm.total)."""
        return self.pbar.total if self.pbar else self._total

    @total.setter
    def total(self, value: Optional[int]) -> None:
        """
        Set total expected iterations and refresh the progress bar.

        Parameters:
        -----------
        value : int, optional
            New total value for the progress bar
        """
        self._total = value
        if self.pbar:
            self.pbar.total = value
            # Force refresh to update the display with new total
            self.pbar.refresh()

    @property
    def elapsed(self) -> float:
        """Seconds since start (proxy to tqdm.elapsed)."""
        if self.pbar and hasattr(self.pbar, "format_dict"):
            return self.pbar.format_dict.get("elapsed", 0.0)
        return time.time() - self.start_time

    def update(
        self,
        n: int = 1,
        postfix: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
    ):
        """
        Update progress by n units.

        Parameters:
        -----------
        n : int
            Number of units to increment progress by
        postfix : dict, optional
            Additional stats to display at the end of the progress bar
        callback : callable, optional
            Function to call after updating progress, receives (current, total)
        """
        if self.track_memory:
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = max(self.peak_memory, current_memory)

            if postfix is None:
                postfix = {}

            postfix["mem"] = f"{current_memory:.1f}MB"

        self.pbar.update(n)

        if postfix:
            self.pbar.set_postfix(**postfix)

        if callback:
            callback(self.pbar.n, self.total)

    def close(self):
        """Close the progress bar and log completion statistics."""
        duration = time.time() - self.start_time
        self.pbar.close()

        if self.track_memory:
            memory_change = self.peak_memory - self.start_memory
            get_logger().info(
                f"Completed: {self.description} in {duration:.2f}s "
                f"(peak memory: {self.peak_memory:.1f}MB, delta: {memory_change:+.1f}MB)"
            )
        else:
            get_logger().info(f"Completed: {self.description} in {duration:.2f}s")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            get_logger().error(
                f"Operation failed: {self.description}. Error: {exc_val}"
            )
        self.close()


# ======= Simple progress bar implementation =======
class SimpleProgressBar(ProgressBase):
    """Simple progress bar for basic tracking needs."""

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        bar_format: Optional[str] = None,
    ):
        """
        Initialize simple progress bar.

        Parameters:
        -----------
        total : int, optional
            Total number of items to process
        description : str
            Description of the current operation
        unit : str
            Unit for the progress bar (e.g., "records", "chunks")
        bar_format : str, optional
            Custom format for the progress bar
        """
        super().__init__(
            total=total,
            description=description,
            unit=unit,
            bar_format=bar_format,
            track_memory=False,
        )


# ======= Standard ProgressBar (consolidates functionality) =======
class ProgressBar(ProgressBase):
    """
    Standard progress bar with memory tracking.
    Compatible with the IO module interface.

    This class consolidates the functionality of the previous ProgressBar
    and SimpleProgressBar to reduce duplication.
    """

    def __init__(
        self,
        total: Optional[int] = None,
        description: str = "Processing",
        unit: str = "items",
        track_memory: bool = True,
        bar_format: Optional[str] = None,
    ):
        """
        Initialize progress bar.

        Parameters:
        -----------
        total : int, optional
            Total number of items to process
        description : str
            Description of the current operation
        unit : str
            Unit for the progress bar (e.g., "records", "chunks")
        track_memory : bool
            Whether to track memory usage (default: True)
        bar_format : str, optional
            Custom format for the progress bar
        """
        super().__init__(
            total=total,
            description=description,
            unit=unit,
            bar_format=bar_format,
            track_memory=track_memory,
        )

    def update(
        self,
        n: int = 1,
        postfix: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Update progress by n units.

        Parameters:
        -----------
        n : int
            Number of units to increment progress by
        postfix : dict, optional
            Additional stats to display at the end of the progress bar
        callback : callable, optional
            Function to call after updating progress (ignored in ProgressBar)
        """
        # Use parent's update method with no callback
        # ProgressBar doesn't support callbacks, so we explicitly pass None
        super().update(n, postfix, callback=None)


# ======= Hierarchical progress tracker =======
class HierarchicalProgressTracker(ProgressBase):
    """
    Enhanced progress tracker with support for nested operations,
    memory tracking, and hierarchical display.
    """

    def __init__(
        self,
        total: int,
        description: str,
        unit: str = "records",
        parent: Optional["HierarchicalProgressTracker"] = None,
        level: int = 0,
        track_memory: bool = True,
        bar_format: Optional[str] = None,
    ):
        """
        Initialize hierarchical progress tracker.

        Parameters:
        -----------
        total : int
            Total number of items to process
        description : str
            Description of the current operation
        unit : str
            Unit for the progress bar (e.g., "records", "batches")
        parent : HierarchicalProgressTracker, optional
            Parent progress tracker for nested operations
        level : int
            Nesting level for hierarchical operations
        track_memory : bool
            Whether to track memory usage during processing
        bar_format : str, optional
            Custom format for the progress bar
        """
        self.parent = parent
        self.level = level
        self.children: List["HierarchicalProgressTracker"] = []

        # Format description with level indentation for nested progress
        prefix = "  " * self.level
        desc = f"{prefix}{description}"

        # Initialize base class
        super().__init__(
            total=total,
            description=description,  # Store original description
            unit=unit,
            track_memory=track_memory,
            bar_format=bar_format,
        )

        # Override pbar with nested position
        if self.pbar:
            self.pbar.close()

        self.pbar = tqdm(
            total=total,
            desc=desc,  # Use formatted description
            unit=unit,
            file=sys.stdout,
            leave=True,
            position=level,
            bar_format=bar_format
            or "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    def create_subtask(
        self,
        total: int,
        description: str,
        unit: str = "items",
        track_memory: Optional[bool] = None,
    ) -> "HierarchicalProgressTracker":
        """
        Create a subtask with its own progress tracking.

        Parameters:
        -----------
        total : int
            Total items in the subtask
        description : str
            Description of the subtask
        unit : str
            Unit for the subtask progress
        track_memory : bool, optional
            Whether to track memory (defaults to parent setting)

        Returns:
        --------
        HierarchicalProgressTracker
            A new progress tracker for the subtask
        """
        if track_memory is None:
            track_memory = self.track_memory

        subtask = HierarchicalProgressTracker(
            total=total,
            description=description,
            unit=unit,
            parent=self,
            level=self.level + 1,
            track_memory=track_memory,
        )

        self.children.append(subtask)
        return subtask

    def close(self):
        """Close the progress bar and log completion statistics."""
        # Close all child progress bars first
        for child in self.children:
            if child.pbar and not child.pbar.disable:
                child.close()

        # Now close this progress bar
        super().close()

        # Update parent progress if this is a subtask
        if self.parent:
            self.parent.update(1)


# ======= Safe context managers =======
@contextmanager
def track_operation_safely(
    description: str,
    total: int,
    unit: str = "items",
    track_memory: bool = True,
    on_error: Optional[Callable[[Exception], None]] = None,
):
    """
    Context manager for tracking an operation with error handling.

    Parameters:
    -----------
    description : str
        Description of the operation
    total : int
        Total number of items to process
    unit : str
        Unit for the progress bar
    track_memory : bool
        Whether to track memory usage
    on_error : callable, optional
        Function to call when an exception occurs

    Yields:
    -------
    HierarchicalProgressTracker
        Progress tracker object
    """
    tracker = HierarchicalProgressTracker(
        total=total, description=description, unit=unit, track_memory=track_memory
    )

    try:
        yield tracker
    except Exception as e:
        get_logger().error(f"Operation failed: {description}. Error: {str(e)}")
        if on_error:
            on_error(e)
        raise
    finally:
        tracker.close()


# ======= Enhanced DataFrame processing =======
def process_dataframe_in_chunks_enhanced(
    df: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], Any],
    description: str,
    chunk_size: int = 10000,
    on_chunk_complete: Optional[Callable[[int, int, Any], None]] = None,
    error_handling: str = "fail",  # Options: "fail", "ignore", "log"
) -> List[Any]:
    """
    Enhanced version: Process a large DataFrame in chunks with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    process_func : callable
        Function to apply to each chunk
    description : str
        Description for the progress bar
    chunk_size : int
        Size of chunks to process at once
    on_chunk_complete : callable, optional
        Function to call when a chunk is processed
    error_handling : str
        How to handle errors: "fail" (raise), "ignore" (skip), "log" (log and skip)

    Returns:
    --------
    list
        List of results from processing each chunk
    """
    total_chunks = int(np.ceil(len(df) / chunk_size))
    results = []
    errors = []

    with track_operation_safely(
        description, total_chunks, unit="chunks", track_memory=True, on_error=None
    ) as tracker:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]

            try:
                # Process the chunk
                result = process_func(chunk)
                results.append(result)

                # Call callback if provided
                if on_chunk_complete:
                    on_chunk_complete(i, total_chunks, result)

            except Exception as e:
                error_msg = f"Error processing chunk {i + 1}/{total_chunks}: {str(e)}"

                if error_handling == "fail":
                    get_logger().error(error_msg)
                    raise
                elif error_handling == "log":
                    get_logger().error(error_msg)
                    errors.append((i, e))
                # "ignore" just skips the error

            # Update progress with memory info
            mem_info = psutil.Process().memory_info()
            tracker.update(
                1,
                {
                    "chunk": f"{i + 1}/{total_chunks}",
                    "mem": f"{mem_info.rss / (1024 * 1024):.1f}MB",
                },
            )

    if errors and error_handling == "log":
        get_logger().warning(f"Completed with {len(errors)} errors")

    return results


def iterate_dataframe_chunks_enhanced(
    df: pd.DataFrame,
    chunk_size: int = 10000,
    description: str = "Processing chunks",
    track_memory: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Enhanced generator that yields chunks of a dataframe with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process in chunks
    chunk_size : int
        Size of chunks to yield
    description : str
        Description for the progress bar
    track_memory : bool
        Whether to track memory usage

    Yields:
    -------
    pd.DataFrame
        Chunk of the original dataframe
    """
    total_chunks = int(np.ceil(len(df) / chunk_size))

    with track_operation_safely(
        description,
        total_chunks,
        unit="chunks",
        track_memory=track_memory,
        on_error=None,
    ) as tracker:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))

            # Get the chunk and update progress
            chunk = df.iloc[start_idx:end_idx].copy()

            # Update tracker with chunk info
            tracker.update(1, {"chunk": f"{i + 1}/{total_chunks}"})

            yield chunk


def process_dataframe_in_parallel_enhanced(
    df: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], Any],
    description: str,
    chunk_size: int = 10000,
    n_jobs: int = -1,
    track_memory: bool = True,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> List[Any]:
    """
    Enhanced function to process a DataFrame in parallel with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    process_func : callable
        Function to apply to each chunk
    description : str
        Description for the progress bar
    chunk_size : int
        Size of chunks to process at once
    n_jobs : int
        Number of parallel jobs (-1 for all processors)
    track_memory : bool
        Whether to track memory usage
    on_error : callable, optional
        Function to call when an exception occurs

    Returns:
    --------
    list
        List of results from processing each chunk
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        get_logger().warning("joblib not installed. Falling back to serial processing.")
        # Use original function for backwards compatibility
        return process_dataframe_in_chunks(df, process_func, description, chunk_size)

    total_chunks = int(np.ceil(len(df) / chunk_size))

    # Create chunks
    chunks = []
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunks.append(df.iloc[start_idx:end_idx])

    # Create a progress bar that will be updated by the parallel executor
    with track_operation_safely(
        description,
        total_chunks,
        unit="chunks",
        track_memory=track_memory,
        on_error=on_error,
    ) as tracker:
        # Define a wrapper that processes a chunk and updates progress
        def process_chunk_with_progress(chunk, chunk_idx):
            result = process_func(chunk)
            # Since tqdm is not thread-safe, we use a workaround
            return result

        # Execute in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_chunk_with_progress)(chunk, i)
            for i, chunk in enumerate(chunks)
        )

        # Update progress bar to completion
        tracker.pbar.update(total_chunks - tracker.pbar.n)

    return results


def multi_stage_process(
    total_stages: int,
    stage_descriptions: List[str],
    stage_weights: Optional[List[float]] = None,
    track_memory: bool = True,
) -> HierarchicalProgressTracker:
    """
    Create a tracker for a multi-stage process.

    Parameters:
    -----------
    total_stages : int
        Total number of processing stages
    stage_descriptions : list
        Descriptions for each stage
    stage_weights : list, optional
        Relative weights for each stage (must sum to 1.0)
    track_memory : bool
        Whether to track memory usage

    Returns:
    --------
    HierarchicalProgressTracker
        Main progress tracker for the entire process
    """
    if stage_weights is None:
        # Equal weights if not specified
        stage_weights = [1.0 / total_stages] * total_stages

    if len(stage_weights) != total_stages or abs(sum(stage_weights) - 1.0) > 1e-10:
        raise ValueError("Stage weights must have same length as stages and sum to 1.0")

    if len(stage_descriptions) != total_stages:
        raise ValueError("Must provide descriptions for all stages")

    # Create master tracker
    master = HierarchicalProgressTracker(
        total=total_stages,
        description="Overall progress",
        unit="stages",
        track_memory=track_memory,
    )

    return master


# ======= Legacy/Backwards Compatibility =======


# Original ProgressTracker for backwards compatibility
@deprecated(alternative="HierarchicalProgressTracker")
class ProgressTracker:
    """
    Comprehensive progress tracking for large data processing operations.
    Supports nested operations, memory tracking, and ETA predictions.

    This class is deprecated, use HierarchicalProgressTracker instead.
    """

    def __init__(
        self,
        total: int,
        description: str,
        unit: str = "records",
        parent: Optional["ProgressTracker"] = None,
        level: int = 0,
        track_memory: bool = True,
    ):
        """
        Initialize progress tracker.

        Parameters:
        -----------
        total : int
            Total number of items to process
        description : str
            Description of the current operation
        unit : str
            Unit for the progress bar (e.g., "records", "batches")
        parent : ProgressTracker, optional
            Parent progress tracker for nested operations
        level : int
            Nesting level for hierarchical operations
        track_memory : bool
            Whether to track memory usage during processing
        """
        # For backwards compatibility
        self.total = total
        self.description = description
        self.unit = unit
        self.parent = parent
        self.level = level
        self.track_memory = track_memory
        self.children = []
        self.start_time = time.time()
        self.start_memory = (
            psutil.Process().memory_info().rss / (1024 * 1024) if track_memory else 0
        )
        self.peak_memory = self.start_memory

        # Format description with level indentation for nested progress
        prefix = "  " * level
        desc = f"{prefix}{description}"

        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stdout,
            leave=True,
            position=level,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

        get_logger().info(f"Started: {description} (total: {total} {unit})")

    # Add proxy properties for compatibility
    @property
    def n(self) -> int:
        """Current counter value (proxy to tqdm.n)."""
        return self.pbar.n if self.pbar else 0

    @property
    def elapsed(self) -> float:
        """Seconds since start."""
        return time.time() - self.start_time

    def update(self, n: int = 1, postfix: Optional[Dict[str, Any]] = None):
        """
        Update progress by n units.

        Parameters:
        -----------
        n : int
            Number of units to increment progress by
        postfix : dict, optional
            Additional stats to display at the end of the progress bar
        """
        if self.track_memory:
            current_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            self.peak_memory = max(self.peak_memory, current_memory)

            if postfix is None:
                postfix = {}

            postfix["mem"] = f"{current_memory:.1f}MB"

        self.pbar.update(n)

        if postfix:
            self.pbar.set_postfix(**postfix)

    def create_subtask(
        self, total: int, description: str, unit: str = "items"
    ) -> "ProgressTracker":
        """
        Create a subtask with its own progress tracking.

        Parameters:
        -----------
        total : int
            Total items in the subtask
        description : str
            Description of the subtask
        unit : str
            Unit for the subtask progress

        Returns:
        --------
        ProgressTracker
            A new progress tracker for the subtask
        """
        subtask = ProgressTracker(
            total=total,
            description=description,
            unit=unit,
            parent=self,
            level=self.level + 1,
            track_memory=self.track_memory,
        )

        self.children.append(subtask)
        return subtask

    def close(self):
        """Close the progress bar and log completion statistics."""
        # Close all child progress bars first
        for child in self.children:
            if child.pbar and not child.pbar.disable:
                child.close()

        duration = time.time() - self.start_time
        self.pbar.close()

        if self.track_memory:
            memory_change = self.peak_memory - self.start_memory
            get_logger().info(
                f"Completed: {self.description} in {duration:.2f}s "
                f"(peak memory: {self.peak_memory:.1f}MB, delta: {memory_change:+.1f}MB)"
            )
        else:
            get_logger().info(f"Completed: {self.description} in {duration:.2f}s")

        # Update parent progress if this is a subtask
        if self.parent:
            self.parent.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Original track_operation for backwards compatibility
@deprecated(alternative="track_operation_safely")
@contextmanager
def track_operation(description: str, total: int, unit: str = "items"):
    """
    Context manager for tracking an operation.

    Parameters:
    -----------
    description : str
        Description of the operation
    total : int
        Total number of items to process
    unit : str
        Unit for the progress bar

    Yields:
    -------
    ProgressTracker
        Progress tracker object
    """
    tracker = ProgressTracker(total=total, description=description, unit=unit)
    try:
        yield tracker
    finally:
        tracker.close()


# Original function for backwards compatibility
@deprecated(alternative="process_dataframe_in_chunks_enhanced")
def process_dataframe_in_chunks(
    df: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], Any],
    description: str,
    chunk_size: int = 10000,
) -> List[Any]:
    """
    Process a large DataFrame in chunks with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    process_func : callable
        Function to apply to each chunk
    description : str
        Description for the progress bar
    chunk_size : int
        Size of chunks to process at once

    Returns:
    --------
    list
        List of results from processing each chunk
    """
    return process_dataframe_in_chunks_enhanced(
        df=df,
        process_func=process_func,
        description=description,
        chunk_size=chunk_size,
        error_handling="fail",
    )


# Original function for backwards compatibility
@deprecated(alternative="iterate_dataframe_chunks_enhanced")
def iterate_dataframe_chunks(
    df: pd.DataFrame, chunk_size: int = 10000, description: str = "Processing chunks"
) -> Iterator[pd.DataFrame]:
    """
    Generator that yields chunks of a dataframe with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process in chunks
    chunk_size : int
        Size of chunks to yield
    description : str
        Description for the progress bar

    Yields:
    -------
    pd.DataFrame
        Chunk of the original dataframe
    """
    yield from iterate_dataframe_chunks_enhanced(df, chunk_size, description)


# Original function for backwards compatibility
@deprecated(alternative="process_dataframe_in_parallel_enhanced")
def process_dataframe_in_parallel(
    df: pd.DataFrame,
    process_func: Callable[[pd.DataFrame], Any],
    description: str,
    chunk_size: int = 10000,
    n_jobs: int = -1,
) -> List[Any]:
    """
    Process a DataFrame in parallel with progress tracking.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to process
    process_func : callable
        Function to apply to each chunk
    description : str
        Description for the progress bar
    chunk_size : int
        Size of chunks to process at once
    n_jobs : int
        Number of parallel jobs (-1 for all processors)

    Returns:
    --------
    list
        List of results from processing each chunk
    """
    return process_dataframe_in_parallel_enhanced(
        df=df,
        process_func=process_func,
        description=description,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        track_memory=True,
        on_error=None,
    )
