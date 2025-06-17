"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Progress Manager
Description: Centralized progress tracking and logging coordination
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides centralized management of progress bars and logging,
ensuring clean display of execution progress without conflicts between
progress indicators and log messages.

Key features:
- Hierarchical progress bars with proper positioning
- Coordinated logging that doesn't break progress displays
- Integration with task reporting
- Support for memory and performance metrics
- Clean exit and resource management
"""

import logging
import os
import sys
import threading
import time
from typing import Dict, Any, Optional, List, Union, Protocol

import tqdm

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # Define psutil as None in the except


class ProgressTrackerProtocol(Protocol):
    """Protocol defining the interface for progress trackers."""

    def update(self, steps: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None:
        """Update progress by specified number of steps."""
        ...

    def set_description(self, description: str) -> None:
        """Update the description of the progress bar."""
        ...

    def set_postfix(self, postfix: Dict[str, Any]) -> None:
        """Set the postfix metrics display."""
        ...

    def close(self, failed: bool = False) -> None:
        """Close the progress bar."""
        ...


class NoOpProgressTracker:
    """
    No-operation progress tracker for quiet mode.

    This class implements the same interface as ProgressTracker but does nothing,
    which is useful for quiet mode or testing.
    """

    def __init__(
            self,
            total: int,
            description: str,
            unit: str = "items",
            position: int = 0,
            leave: bool = True,
            parent: Optional[Any] = None,
            color: Optional[str] = None
    ):
        """
        Initialize no-op progress tracker.

        Args:
            total: Total number of steps (unused)
            description: Description of the operation (unused)
            unit: Unit of progress (unused)
            position: Fixed position on screen (unused)
            leave: Whether to leave the progress bar after completion (unused)
            parent: Parent progress tracker (unused)
            color: Color of the progress bar (unused)
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.position = position
        self.leave = leave
        self.parent = parent
        self.color = color
        self.children = []

        # Start time and memory tracking still works in quiet mode
        self.start_time = time.time()
        self.start_memory = self._get_current_memory()
        self.peak_memory = self.start_memory

        # Custom metrics still collected in quiet mode
        self.metrics: Dict[str, Any] = {}

    def update(
            self,
            steps: int = 1,
            postfix: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress (no-op version).

        Args:
            steps: Number of steps completed
            postfix: Dictionary of metrics to display after the progress bar
        """
        # Still track metrics even in quiet mode
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)

        # Add steps to metrics
        if "current" not in self.metrics:
            self.metrics["current"] = 0
        self.metrics["current"] += steps

    def set_description(self, description: str) -> None:
        """
        Update the description (no-op version).

        Args:
            description: New description text
        """
        self.description = description

    def set_postfix(self, postfix: Dict[str, Any]) -> None:
        """
        Set the postfix metrics (no-op version).

        Args:
            postfix: Dictionary of metrics to display
        """
        pass

    def close(self, failed: bool = False) -> None:
        """
        Close the progress tracker (no-op version).

        Args:
            failed: Whether the operation failed
        """
        # Calculate execution time
        execution_time = time.time() - self.start_time

        # Update metrics
        self.metrics.update({
            'execution_time': execution_time,
            'peak_memory_mb': self.peak_memory,
            'memory_delta_mb': self.peak_memory - self.start_memory,
            'items_per_second': self.metrics.get("current", 0) / execution_time if execution_time > 0 else 0,
            'failed': failed
        })

        # Update parent if exists
        if self.parent:
            try:
                self.parent.update(1)
            except Exception:
                # Parent may have been closed already
                pass

    def _get_current_memory(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def __enter__(self) -> 'NoOpProgressTracker':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close(failed=exc_type is not None)


class ProgressTracker:
    """
    Progress tracker for individual operations with fixed positioning.

    This class wraps tqdm with additional functionality for hierarchical
    display, metrics collection, and proper positioning.
    """

    def __init__(
            self,
            total: int,
            description: str,
            unit: str = "items",
            position: int = 0,
            leave: bool = True,
            parent: Optional['ProgressTracker'] = None,
            color: Optional[str] = None,
            disable: bool = False
    ):
        """
        Initialize progress tracker.

        Args:
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress (e.g., "items", "records")
            position: Fixed position on screen (0 = top)
            leave: Whether to leave the progress bar after completion
            parent: Parent progress tracker (for hierarchical display)
            color: Color of the progress bar (None for default)
            disable: Whether to disable the progress bar display
        """
        self.total = total
        self.description = description
        self.unit = unit
        self.position = position
        self.leave = leave
        self.parent = parent
        self.color = color
        self.children: List['ProgressTracker'] = []
        self.disabled = disable

        # Start time and memory tracking
        self.start_time = time.time()
        self.start_memory = self._get_current_memory()
        self.peak_memory = self.start_memory

        # Custom metrics
        self.metrics: Dict[str, Any] = {}

        # Create progress bar with fixed positioning
        self.pbar = tqdm.tqdm(
            total=total,
            desc=description,
            unit=unit,
            position=position,
            leave=leave,
            file=sys.stdout,
            colour=color,
            disable=disable
        )

        # Register with parent if exists
        if parent is not None:
            parent.children.append(self)

    def update(
            self,
            steps: int = 1,
            postfix: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress by specified number of steps.

        Args:
            steps: Number of steps completed
            postfix: Dictionary of metrics to display after the progress bar
        """
        # Track memory metrics regardless of whether bar is visible
        current_memory = self._get_current_memory()
        self.peak_memory = max(self.peak_memory, current_memory)

        # Create postfix if not provided
        if postfix is None:
            postfix = {}

        # Add memory info to postfix if significant
        if 'mem' not in postfix and current_memory > 1.0:  # Only add if > 1MB
            postfix['mem'] = f"{current_memory:.1f}MB"

        # Update progress bar
        self.pbar.update(steps)

        # Set postfix if not empty
        if postfix:
            self.pbar.set_postfix(**postfix)

    def set_description(self, description: str) -> None:
        """
        Update the description of the progress bar.

        Args:
            description: New description text
        """
        self.description = description
        self.pbar.set_description(description)

    def set_postfix(self, postfix: Dict[str, Any]) -> None:
        """
        Set the postfix metrics display.

        Args:
            postfix: Dictionary of metrics to display
        """
        self.pbar.set_postfix(**postfix)

    def close(self, failed: bool = False) -> None:
        """
        Close the progress bar and compute final metrics.

        Args:
            failed: Whether the operation failed
        """
        # Close all child progress bars first, with a copy of the list to safely iterate
        for child in list(self.children):
            child.close(failed=failed)
            if child in self.children:  # Check if still in list after closing
                self.children.remove(child)

        # Calculate execution time
        execution_time = time.time() - self.start_time

        # Update metrics
        current = getattr(self.pbar, "n", 0)
        self.metrics.update({
            'execution_time': execution_time,
            'peak_memory_mb': self.peak_memory,
            'memory_delta_mb': self.peak_memory - self.start_memory,
            'items_per_second': current / execution_time if execution_time > 0 else 0,
            'failed': failed
        })

        # Change color if failed
        if failed and self.pbar and not self.disabled:
            try:
                # Try to set color to red, using bar_format if colour attribute fails
                if hasattr(self.pbar, 'colour'):
                    # First attempt with colour attribute
                    self.pbar.colour = 'red'
                else:
                    # Fallback - use ANSI codes directly in bar_format
                    # This is more cross-platform compatible
                    red_format = "\033[91m{l_bar}{bar:40}\033[0m{r_bar}"
                    self.pbar.bar_format = red_format
            except Exception:
                # Ignore any errors in changing color
                pass

        # Close the progress bar
        if self.pbar:
            self.pbar.close()

        # Update parent progress if exists
        if self.parent:
            try:
                self.parent.update(1)
            except Exception:
                # Parent may have been closed already
                pass

    def clear(self) -> None:
        """
        Clear the progress bar from display.
        This is useful when temporarily hiding the bar to display messages.
        """
        if self.pbar and not self.disabled:
            self.pbar.clear()

    def refresh(self) -> None:
        """
        Redraw the progress bar.
        This is useful after temporarily clearing the bar.
        """
        if self.pbar and not self.disabled:
            self.pbar.refresh()

    def _get_current_memory(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def __enter__(self) -> 'ProgressTracker':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close(failed=exc_type is not None)


class TaskProgressManager:
    """
    Centralized manager for task progress and logging coordination.

    This class coordinates progress display and logging to ensure they
    don't interfere with each other, creating a clean user experience.
    """

    def __init__(
            self,
            task_id: str,
            task_type: str,
            logger: logging.Logger,
            reporter: Optional[Any] = None,  # TaskReporter type, avoid circular import
            total_operations: int = 0,
            quiet: bool = False
    ):
        """
        Initialize progress manager.

        Args:
            task_id: Task identifier
            task_type: Type of task
            logger: Logger for the task
            reporter: Task reporter for metrics
            total_operations: Total number of operations (if known)
            quiet: Whether to disable progress bars
        """
        self.task_id = task_id
        self.task_type = task_type
        self.logger = logger
        self.reporter = reporter
        self.total_operations = total_operations
        self.quiet = quiet

        # Thread safety
        self.lock = threading.Lock()

        # Ensure logger doesn't write to stdout
        self._check_logger_handlers()

        # Task state tracking
        self.operations_completed = 0
        self.active_operations: Dict[str, Union[ProgressTracker, NoOpProgressTracker]] = {}
        self.start_time = time.time()
        self.peak_memory = self._get_current_memory()

        # Main progress bar for overall task progress
        self.main_progress = None
        if not quiet and total_operations > 0:
            self.main_progress = ProgressTracker(
                total=total_operations,
                description=f"Task: {task_id} ({task_type})",
                unit="operations",
                position=0,
                leave=True
            )

    def set_total_operations(self, total: int) -> None:
        """
        Set the total number of operations for the task.

        This method is expected by BaseTask to update the total number of operations
        for progress tracking. When using the BaseTask.run() method, it sets the total
        operations equal to the length of the operations list.

        Args:
            total: Total number of operations
        """
        with self.lock:
            self.logger.debug(f"Setting total operations to {total}")
            self.total_operations = total

            # Update main progress bar if it exists
            if self.main_progress and hasattr(self.main_progress, 'pbar') and hasattr(self.main_progress.pbar, 'total'):
                try:
                    self.main_progress.pbar.total = total
                    # Also refresh the display
                    self.main_progress.pbar.refresh()
                except Exception as e:
                    # Handle any errors gracefully
                    self.logger.warning(f"Could not update progress bar total: {e}")

    def increment_total_operations(self, count: int = 1) -> None:
        """
        Increment the total operations count.

        This method is expected by BaseTask in the add_operation() method
        to increment the total operation count when a new operation is added.

        Args:
            count: Number of operations to add (default: 1)
        """
        with self.lock:
            self.logger.debug(f"Incrementing total operations by {count}")
            self.total_operations += count

            # Update main progress bar if it exists
            if self.main_progress and hasattr(self.main_progress, 'pbar') and hasattr(self.main_progress.pbar, 'total'):
                try:
                    self.main_progress.pbar.total += count
                    # Also refresh the display
                    self.main_progress.pbar.refresh()
                except Exception as e:
                    # Handle any errors gracefully
                    self.logger.warning(f"Could not update progress bar total: {e}")

    def _check_logger_handlers(self) -> None:
        """
        Check logger handlers to prevent conflicts with progress bars.
        Warns if a handler might write to stdout.
        """
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
                # This could conflict with progress bars
                self.logger.warning(
                    "Logger has a StreamHandler writing to stdout, which may conflict with progress bars. "
                    "Consider using stderr instead."
                )

    def start_operation(
            self,
            name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ) -> Union[ProgressTracker, NoOpProgressTracker]:
        """
        Start tracking a new operation.

        Args:
            name: Operation name (unique identifier)
            total: Total number of steps in the operation
            description: Description of the operation (defaults to name)
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion

        Returns:
            Progress tracker for the operation
        """
        with self.lock:
            # Create operation description if not provided
            if description is None:
                description = f"Operation: {name}"

            # Check for empty work (total <= 0)
            if total <= 0:
                self.log_debug(f"No items to process for '{name}', using no-op progress tracker")
                # Return a no-op tracker for empty operations
                tracker = NoOpProgressTracker(
                    total=total,
                    description=description,
                    unit=unit,
                    position=0,
                    leave=leave,
                    parent=self.main_progress
                )
                self.active_operations[name] = tracker
                return tracker

            # Skip visual progress bars if in quiet mode
            if self.quiet:
                # Return a no-op tracker that maintains the interface
                tracker = NoOpProgressTracker(
                    total=total,
                    description=description,
                    unit=unit,
                    position=0,
                    leave=leave,
                    parent=self.main_progress
                )
                self.active_operations[name] = tracker
                return tracker

            # Calculate position based on number of active operations
            position = len(self.active_operations) + 1  # +1 because main_progress is at position 0

            # Create progress tracker
            progress = ProgressTracker(
                total=total,
                description=description,
                unit=unit,
                position=position,
                leave=leave,
                parent=self.main_progress
            )

            # Register active operation
            self.active_operations[name] = progress

            # Log operation start
            self.log_info(f"Starting operation: {name}")

            return progress

    def update_operation(
            self,
            name: str,
            steps: int = 1,
            postfix: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update progress of an operation.

        Args:
            name: Operation name
            steps: Number of steps completed
            postfix: Additional metrics to display
        """
        with self.lock:
            # Update operation progress if operation exists
            if name in self.active_operations:
                self.active_operations[name].update(steps, postfix)

    def complete_operation(
            self,
            name: str,
            success: bool = True,
            metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Mark an operation as completed.

        Args:
            name: Operation name
            success: Whether the operation completed successfully
            metrics: Final metrics for the operation
        """
        with self.lock:
            # Update operation counter
            self.operations_completed += 1

            # Log completion
            level = logging.INFO if success else logging.ERROR
            status = "completed successfully" if success else "failed"
            self.log_message(level, f"Operation {name} {status}")

            # Get metrics from tracker if available
            operation_metrics = {}
            if metrics:
                operation_metrics.update(metrics)

            if name in self.active_operations:
                # Get tracker
                tracker = self.active_operations[name]

                # Update metrics if not already provided
                if metrics is None:
                    operation_metrics = tracker.metrics
                else:
                    # Merge tracker metrics with provided metrics
                    tracker.metrics.update(metrics)
                    operation_metrics = tracker.metrics

                # Close progress tracker
                tracker.close(failed=not success)

                # Remove from active operations
                del self.active_operations[name]

            # Add to reporter if available
            if self.reporter:
                status_name = "success" if success else "error"
                self.reporter.add_operation(
                    name=f"Complete {name}",
                    status=status_name,
                    details=operation_metrics
                )

            # Update task peak memory tracking
            current_memory = self._get_current_memory()
            self.peak_memory = max(self.peak_memory, current_memory)

            # Update main progress bar if it exists
            if self.main_progress and hasattr(self.main_progress, 'pbar'):
                try:
                    # Update progress of main bar
                    self.main_progress.update(1)
                except Exception as e:
                    # Handle any errors gracefully
                    self.logger.debug(f"Could not update main progress: {e}")

    def log_message(
            self,
            level: int,
            message: str,
            preserve_progress: bool = True
    ) -> None:
        """
        Log a message without breaking progress bars.

        Args:
            level: Logging level
            message: Message to log
            preserve_progress: Whether to preserve progress bars after logging
        """
        with self.lock:
            # First, log through the logger (this goes to file)
            self.logger.log(level, message)

            # Then, display on console without breaking progress bars
            if self.main_progress and not self.quiet:
                # First clear progress bars if not preserving them
                if not preserve_progress:
                    self.main_progress.clear()
                    for tracker in self.active_operations.values():
                        if hasattr(tracker, 'clear'):
                            tracker.clear()

                # Use tqdm.write to preserve progress bars
                level_name = logging.getLevelName(level)
                self.main_progress.pbar.write(f"[{level_name}] {message}")

                # Refresh progress bars if not preserving them
                if not preserve_progress:
                    for tracker in self.active_operations.values():
                        if hasattr(tracker, 'refresh'):
                            tracker.refresh()
                    self.main_progress.refresh()
            else:
                # Direct output to stderr if no progress bars or quiet mode
                if level >= logging.WARNING:  # Only print warnings and errors to stderr in quiet mode
                    print(f"[{logging.getLevelName(level)}] {message}", file=sys.stderr)

    def log_info(self, message: str) -> None:
        """Convenience method for logging info messages."""
        self.log_message(logging.INFO, message)

    def log_warning(self, message: str) -> None:
        """Convenience method for logging warning messages."""
        self.log_message(logging.WARNING, message)

    def log_error(self, message: str) -> None:
        """Convenience method for logging error messages."""
        self.log_message(logging.ERROR, message)

    def log_debug(self, message: str) -> None:
        """Convenience method for logging debug messages."""
        self.log_message(logging.DEBUG, message)

    def log_critical(self, message: str, preserve_progress: bool = False) -> None:
        """
        Convenience method for logging critical messages.

        By default, critical messages clear progress bars to ensure visibility.

        Args:
            message: Critical message to log
            preserve_progress: Whether to preserve progress bars after logging
        """
        self.log_message(logging.CRITICAL, message, preserve_progress=preserve_progress)

    def create_operation_context(
            self,
            name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ) -> 'ProgressContext':
        """
        Create a context manager for an operation.

        Args:
            name: Operation name
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion

        Returns:
            Context manager for the operation
        """
        # Log a warning if total is zero or negative
        if total <= 0:
            self.log_debug(f"Creating progress context for '{name}' with total={total} (no items to process)")

        return ProgressContext(
            progress_manager=self,
            operation_name=name,
            total=total,
            description=description,
            unit=unit,
            leave=leave
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get overall metrics for the task.

        Returns:
            Dictionary of task metrics
        """
        execution_time = time.time() - self.start_time
        current_memory = self._get_current_memory()

        # Calculate operations per second, avoiding division by zero
        ops_per_second = 0
        if execution_time > 0:
            ops_per_second = self.operations_completed / execution_time

        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'operations_completed': self.operations_completed,
            'operations_total': self.total_operations,
            'execution_time': execution_time,
            'operations_per_second': ops_per_second,
            'peak_memory_mb': self.peak_memory,
            'current_memory_mb': current_memory,
            'memory_delta_mb': current_memory - self._get_initial_memory()
        }

    def close(self) -> None:
        """Close all progress bars and release resources."""
        with self.lock:
            # Close all active operations
            for name, progress in list(self.active_operations.items()):
                try:
                    progress.close()
                except Exception as e:
                    self.logger.debug(f"Error closing progress for {name}: {e}")

            # Clear active operations
            self.active_operations.clear()

            # Close main progress bar
            if self.main_progress:
                try:
                    self.main_progress.close()
                except Exception as e:
                    self.logger.debug(f"Error closing main progress bar: {e}")
                self.main_progress = None

    def _get_current_memory(self) -> float:
        """
        Get current memory usage in MB.

        Returns:
            Memory usage in MB
        """
        if PSUTIL_AVAILABLE:
            try:
                return psutil.Process().memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    def _get_initial_memory(self) -> float:
        """
        Get initial memory usage at manager creation.

        Returns:
            Initial memory usage in MB
        """
        # Start time is approximately when the manager was created
        # We can estimate initial memory as 0 if not available
        return 0.0

    def __enter__(self) -> 'TaskProgressManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        # Log any exception
        if exc_type:
            self.log_error(f"Task failed with exception: {exc_val}")

        # Close all progress bars
        self.close()


class ProgressContext:
    """
    Context manager for operation execution with progress tracking.

    This class provides a convenient way to track progress of an operation
    using the context manager pattern (with statement).
    """

    def __init__(
            self,
            progress_manager: TaskProgressManager,
            operation_name: str,
            total: int,
            description: Optional[str] = None,
            unit: str = "items",
            leave: bool = False
    ):
        """
        Initialize progress context.

        Args:
            progress_manager: Task progress manager
            operation_name: Name of the operation
            total: Total number of steps
            description: Description of the operation
            unit: Unit of progress
            leave: Whether to leave the progress bar after completion
        """
        self.progress_manager = progress_manager
        self.operation_name = operation_name
        self.total = max(0, total)  # Ensure non-negative total
        self.description = description
        self.unit = unit
        self.leave = leave
        self.tracker = None
        self.metrics = {}
        self.empty_operation = (total <= 0)

    def __enter__(self) -> Union[ProgressTracker, NoOpProgressTracker]:
        """
        Start tracking operation progress.

        Returns:
            Progress tracker for the operation
        """
        # Check for empty work (total <= 0)
        if self.empty_operation:
            # Log that we're skipping progress tracking for this empty operation
            if self.progress_manager and self.progress_manager.logger:
                self.progress_manager.logger.debug(
                    f"No items to process for '{self.operation_name}', using no-op progress tracker"
                )

            # Create a no-op tracker instead of a real progress bar
            self.tracker = NoOpProgressTracker(
                total=0,
                description=self.description or f"Operation: {self.operation_name}",
                unit=self.unit,
                position=0,
                leave=self.leave
            )
            return self.tracker

        # Normal case: create a real progress tracker
        self.tracker = self.progress_manager.start_operation(
            name=self.operation_name,
            total=self.total,
            description=self.description,
            unit=self.unit,
            leave=self.leave
        )
        return self.tracker

    def __exit__(
            self,
            exc_type,
            exc_val,
            exc_tb
    ) -> None:
        """
        Complete operation tracking.

        Args:
            exc_type: Exception type if an exception occurred
            exc_val: Exception value if an exception occurred
            exc_tb: Exception traceback if an exception occurred
        """
        # Determine if operation was successful
        success = exc_type is None

        # Collect metrics from tracker if available
        if self.tracker:
            self.metrics.update(self.tracker.metrics)

        # Add exception info to metrics if failed
        if not success and exc_type and exc_val:
            self.metrics['error_type'] = exc_type.__name__
            self.metrics['error_message'] = str(exc_val)

        # Complete operation (differently based on empty status)
        if not self.empty_operation:
            try:
                self.progress_manager.complete_operation(
                    name=self.operation_name,
                    success=success,
                    metrics=self.metrics
                )
            except Exception as e:
                # Log error but don't raise to ensure context exit completes
                if self.progress_manager and self.progress_manager.logger:
                    self.progress_manager.logger.error(f"Error completing operation: {e}")
        elif self.tracker:
            # Just close the no-op tracker without registering completion
            try:
                self.tracker.close(failed=not success)
            except Exception as e:
                # Log error but don't raise
                if self.progress_manager and self.progress_manager.logger:
                    self.progress_manager.logger.error(f"Error closing no-op tracker: {e}")


def create_task_progress_manager(
        task_id: str,
        task_type: str,
        logger: logging.Logger,
        reporter: Optional[Any] = None,  # TaskReporter type, avoid circular import
        total_operations: int = 0,
        quiet: Optional[bool] = None
) -> TaskProgressManager:
    """
    Create a task progress manager with auto-detection of quiet mode.

    This is a convenience function for creating a TaskProgressManager
    with optional auto-detection of quiet mode based on the environment.

    Args:
        task_id: Task identifier
        task_type: Type of task
        logger: Logger for the task
        reporter: Task reporter for metrics
        total_operations: Total number of operations (if known)
        quiet: Whether to disable progress bars (auto-detected if None)

    Returns:
        Task progress manager
    """
    # Auto-detect quiet mode if not specified
    if quiet is None:
        # Detect if running in non-interactive environment
        # This handles pipes, redirects, and non-terminal environments
        quiet = not sys.stdout.isatty()

        # Also check if we're running in a CI environment (common quiet mode indicator)
        if "CI" in os.environ or "CONTINUOUS_INTEGRATION" in os.environ:
            quiet = True

    return TaskProgressManager(
        task_id=task_id,
        task_type=task_type,
        logger=logger,
        reporter=reporter,
        total_operations=total_operations,
        quiet=quiet
    )