"""
HHA (HeadHunter Anonymization) - Progress Tracking
--------------------------------------------------
This module provides robust progress tracking utilities for processing
very large datasets in the context of privacy-preserving transformations.

Features:
- Real-time progress visualization with ETA
- Multi-stage process tracking
- Memory-efficient processing for datasets with millions of records
- Custom logging formats for different operation types
- Support for nested operations with hierarchical progress reporting

(C) 2025 BDA

Author: V.Khvatov
"""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Callable, Iterator

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("hha_processing.log")
    ]
)

logger = logging.getLogger("hha")


class ProgressBar:
    """
    Simple progress bar wrapper using tqdm.
    Compatible with the IO module interface.
    """

    def __init__(self,
                 total: Optional[int] = None,
                 description: str = "Processing",
                 unit: str = "items"):
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
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.start_time = time.time()

        self.pbar = tqdm(
            total=total,
            desc=description,
            unit=unit,
            file=sys.stdout,
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        logger.info(f"Started: {description}" + (f" (total: {total} {unit})" if total else ""))

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
        self.pbar.update(n)

        if postfix:
            self.pbar.set_postfix(**postfix)

    def close(self):
        """Close the progress bar and log completion statistics."""
        duration = time.time() - self.start_time
        self.pbar.close()

        logger.info(f"Completed: {self.description} in {duration:.2f}s")


class ProgressTracker:
    """
    Comprehensive progress tracking for large data processing operations.
    Supports nested operations, memory tracking, and ETA predictions.
    """

    def __init__(self,
                 total: int,
                 description: str,
                 unit: str = "records",
                 parent: Optional['ProgressTracker'] = None,
                 level: int = 0,
                 track_memory: bool = True):
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
        self.total = total
        self.description = description
        self.unit = unit
        self.parent = parent
        self.level = level
        self.track_memory = track_memory
        self.start_time = time.time()
        self.children: List['ProgressTracker'] = []
        self.peak_memory = 0
        self.start_memory = psutil.Process().memory_info().rss / (1024 * 1024) if track_memory else 0

        # Format description with level indentation for nested progress
        prefix = "  " * self.level
        desc = f"{prefix}{description}"

        self.pbar = tqdm(
            total=total,
            desc=desc,
            unit=unit,
            file=sys.stdout,
            leave=True,
            position=level,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        logger.info(f"Started: {description} (total: {total} {unit})")

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

            postfix['mem'] = f"{current_memory:.1f}MB"

        self.pbar.update(n)

        if postfix:
            self.pbar.set_postfix(**postfix)

    def create_subtask(self, total: int, description: str, unit: str = "items") -> 'ProgressTracker':
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
            track_memory=self.track_memory
        )

        self.children.append(subtask)
        return subtask

    def close(self):
        """Close the progress bar and log completion statistics."""
        duration = time.time() - self.start_time
        self.pbar.close()

        # Calculate and log memory usage if tracking is enabled
        if self.track_memory:
            memory_change = self.peak_memory - self.start_memory
            logger.info(
                f"Completed: {self.description} in {duration:.2f}s "
                f"(peak memory: {self.peak_memory:.1f}MB, delta: {memory_change:+.1f}MB)"
            )
        else:
            logger.info(f"Completed: {self.description} in {duration:.2f}s")

        # Update parent progress if this is a subtask
        if self.parent:
            self.parent.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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


def process_dataframe_in_chunks(
        df: pd.DataFrame,
        process_func: Callable[[pd.DataFrame], Any],
        description: str,
        chunk_size: int = 10000
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
    total_chunks = int(np.ceil(len(df) / chunk_size))
    results = []

    with track_operation(description, total_chunks, unit="chunks") as tracker:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]

            # Process the chunk
            result = process_func(chunk)
            results.append(result)

            # Update progress with memory info
            mem_info = psutil.Process().memory_info()
            tracker.update(1, {"chunk": f"{i + 1}/{total_chunks}", "mem": f"{mem_info.rss / (1024 * 1024):.1f}MB"})

    return results


def iterate_dataframe_chunks(
        df: pd.DataFrame,
        chunk_size: int = 10000,
        description: str = "Processing chunks"
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
    total_chunks = int(np.ceil(len(df) / chunk_size))

    with track_operation(description, total_chunks, unit="chunks") as tracker:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))

            # Get the chunk and update progress
            chunk = df.iloc[start_idx:end_idx].copy()

            # Update tracker with chunk info
            tracker.update(1, {"chunk": f"{i + 1}/{total_chunks}"})

            yield chunk


def multi_stage_process(
        total_stages: int,
        stage_descriptions: List[str],
        stage_weights: Optional[List[float]] = None
) -> 'ProgressTracker':
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

    Returns:
    --------
    ProgressTracker
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
    master = ProgressTracker(
        total=total_stages,
        description="Overall progress",
        unit="stages"
    )

    return master