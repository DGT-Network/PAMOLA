"""
PAMOLA.CORE - Privacy-Preserving AI Data Processors
----------------------------------------------------
Module: Operation Executor
Description: Task operation execution with retry capabilities
Author: PAMOLA Core Team
Created: 2025
License: BSD 3-Clause

This module provides functionality for executing operations with retry capabilities,
handling temporary failures gracefully with exponential backoff, and supporting
selective retry based on exception types.

Key features:
- Operation execution with configurable retry strategies
- Exponential backoff for transient failures
- Selective retry based on exception types
- Progress tracking during execution
- Integration with task reporting
"""

import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional, Type, Set, Callable

from pamola_core.utils.ops.op_base import BaseOperation
from pamola_core.utils.ops.op_result import OperationResult, OperationStatus
from pamola_core.utils.progress import ProgressTracker
from pamola_core.utils.tasks.task_reporting import TaskReporter

# Define a generic exception type for better type annotations
ExceptionType = Type[BaseException]


class ExecutionError(Exception):
    """Base exception for operation execution errors."""
    pass


class MaxRetriesExceededError(ExecutionError):
    """Exception raised when maximum retry attempts are reached."""
    pass


class NonRetriableError(ExecutionError):
    """Exception raised for errors that should not be retried."""
    pass


class TaskOperationExecutor:
    """
    Executor for task operations with retry capabilities.

    This class handles the execution of operations within a task, with support for
    retry logic, progress tracking, and result collection. It provides configurable
    retry strategies including exponential backoff and selective retry based on
    exception types.
    """

    # Default set of exception types that are generally safe to retry
    DEFAULT_RETRIABLE_EXCEPTIONS = {
        ConnectionError,  # Network-related errors
        TimeoutError,  # Timeout errors
        IOError,  # I/O-related errors (file access, etc.)
        BrokenPipeError,  # Pipe-related errors
        ConnectionResetError,  # Connection reset errors
        ConnectionAbortedError,  # Connection aborted errors
    }

    # Exception types that should never be retried
    NEVER_RETRY_EXCEPTIONS = {
        KeyboardInterrupt,  # User interrupted execution
        SystemExit,  # System exit requested
        MemoryError,  # Out of memory
        NonRetriableError,  # Explicitly marked as non-retriable
        SyntaxError,  # Syntax errors in code
        NameError,  # Undefined name
        TypeError,  # Type errors
        ValueError,  # Invalid value
    }

    def __init__(self,
                 task_config: Any,
                 logger: logging.Logger,
                 reporter: Optional[TaskReporter] = None,
                 default_max_retries: int = 3,
                 default_backoff_factor: float = 2.0,
                 default_initial_wait: float = 1.0,
                 default_max_wait: float = 60.0,
                 default_jitter: bool = True):
        """
        Initialize the operation executor.

        Args:
            task_config: Task configuration object
            logger: Logger for operation execution
            reporter: Task reporter for tracking operation results
            default_max_retries: Default maximum retry attempts
            default_backoff_factor: Default backoff factor for exponential backoff
            default_initial_wait: Default initial wait time in seconds
            default_max_wait: Default maximum wait time in seconds
            default_jitter: Whether to add jitter to wait times
        """
        self.config = task_config
        self.logger = logger
        self.reporter = reporter

        # Default retry settings
        self.default_max_retries = default_max_retries
        self.default_backoff_factor = default_backoff_factor
        self.default_initial_wait = default_initial_wait
        self.default_max_wait = default_max_wait
        self.default_jitter = default_jitter

        # Custom retriable exceptions (can be extended)
        self.retriable_exceptions: Set[ExceptionType] = set(self.DEFAULT_RETRIABLE_EXCEPTIONS)

        # Track execution statistics
        self.execution_stats = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "retried_operations": 0,
            "total_retries": 0,
        }

    def add_retriable_exception(self, exception_type: ExceptionType) -> None:
        """
        Add an exception type to the set of retriable exceptions.

        Args:
            exception_type: The exception type to add
        """
        if exception_type in self.NEVER_RETRY_EXCEPTIONS:
            self.logger.warning(
                f"Exception type {exception_type.__name__} is in NEVER_RETRY_EXCEPTIONS "
                f"and cannot be added to retriable exceptions."
            )
            return

        self.retriable_exceptions.add(exception_type)
        self.logger.debug(f"Added {exception_type.__name__} to retriable exceptions")

    def remove_retriable_exception(self, exception_type: ExceptionType) -> None:
        """
        Remove an exception type from the set of retriable exceptions.

        Args:
            exception_type: The exception type to remove
        """
        if exception_type in self.retriable_exceptions:
            self.retriable_exceptions.remove(exception_type)
            self.logger.debug(f"Removed {exception_type.__name__} from retriable exceptions")

    def is_retriable_error(self, exception: Exception) -> bool:
        """
        Determine if an exception should trigger a retry.

        Args:
            exception: The exception to check

        Returns:
            True if the exception is retriable, False otherwise
        """
        # Check if the exception is in the never-retry list
        for exc_type in self.NEVER_RETRY_EXCEPTIONS:
            if isinstance(exception, exc_type):
                return False

        # Check if the exception is in the retriable list
        for exc_type in self.retriable_exceptions:
            if isinstance(exception, exc_type):
                return True

        # Check if the exception has a 'retriable' attribute
        if hasattr(exception, 'retriable'):
            return bool(exception.retriable)

        # Default to non-retriable for unknown exceptions
        return False

    def execute_operation(self,
                          operation: BaseOperation,
                          params: Dict[str, Any],
                          progress_tracker: Optional[ProgressTracker] = None) -> OperationResult:
        """
        Execute a single operation without retry logic.

        Args:
            operation: The operation to execute
            params: Parameters for the operation
            progress_tracker: Progress tracker for the operation

        Returns:
            OperationResult containing the execution result

        Raises:
            Exception: Any exception raised by the operation is propagated up
        """
        operation_name = operation.__class__.__name__
        self.logger.info(f"Executing operation: {operation_name}")

        # Update execution statistics
        self.execution_stats["total_operations"] += 1

        try:
            # Run the operation - let exceptions propagate to the caller
            result = operation.run(**params)

            # Update execution statistics
            if result.status == OperationStatus.SUCCESS:
                self.execution_stats["successful_operations"] += 1
            else:
                self.execution_stats["failed_operations"] += 1

            # Log the result
            self.logger.info(
                f"Operation {operation_name} completed with status: {result.status.name}"
            )

            return result
        except Exception as e:
            # Immediately re-raise KeyboardInterrupt to allow process termination
            if isinstance(e, KeyboardInterrupt):
                raise

            # For other exceptions, update stats and re-raise
            self.execution_stats["failed_operations"] += 1
            self.logger.error(f"Operation {operation_name} failed with exception: {str(e)}")
            raise

    def _make_error_result(self, exception: Exception, execution_time: float,
                           additional_message: Optional[str] = None) -> OperationResult:
        """
        Create an error OperationResult from an exception.

        Args:
            exception: The exception that caused the error
            execution_time: Execution time in seconds
            additional_message: Additional message to append to error_message

        Returns:
            OperationResult with ERROR status and appropriate error information
        """
        error_message = str(exception)
        if additional_message:
            error_message += f" {additional_message}"

        # Format traceback if enabled
        error_trace = self._format_exception(exception) if getattr(self.config, 'store_traceback', True) else ""

        return OperationResult(
            status=OperationStatus.ERROR,
            error_message=error_message,
            execution_time=execution_time,
            error_trace=error_trace
        )

    def execute_with_retry(self,
                           operation: BaseOperation,
                           params: Dict[str, Any],
                           max_retries: Optional[int] = None,
                           backoff_factor: Optional[float] = None,
                           initial_wait: Optional[float] = None,
                           max_wait: Optional[float] = None,
                           jitter: Optional[bool] = None,
                           progress_tracker: Optional[ProgressTracker] = None,
                           on_retry: Optional[Callable[[Exception, int, float], None]] = None) -> OperationResult:
        """
        Execute an operation with retry logic.

        Args:
            operation: The operation to execute
            params: Parameters for the operation
            max_retries: Maximum number of retry attempts
            backoff_factor: Backoff factor for exponential backoff
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds
            jitter: Whether to add jitter to wait times
            progress_tracker: Progress tracker for the operation
            on_retry: Callback function called before each retry attempt

        Returns:
            OperationResult containing the execution result

        Raises:
            MaxRetriesExceededError: If maximum retry attempts are reached
            NonRetriableError: If an exception is raised that should not be retried
            KeyboardInterrupt: If the user interrupts execution with Ctrl+C

        Note:
            The execution_time in the returned OperationResult represents the total time
            across all retry attempts, not just the final successful attempt.
        """
        # Use provided values or defaults
        max_retries = max_retries if max_retries is not None else self.default_max_retries
        backoff_factor = backoff_factor if backoff_factor is not None else self.default_backoff_factor
        initial_wait = initial_wait if initial_wait is not None else self.default_initial_wait
        max_wait = max_wait if max_wait is not None else self.default_max_wait
        jitter = jitter if jitter is not None else self.default_jitter

        operation_name = operation.__class__.__name__
        total_execution_time = 0.0

        for attempt in range(max_retries + 1):
            attempt_start_time = time.time()

            try:
                # Execute the operation
                if attempt == 0:
                    self.logger.info(f"Executing operation: {operation_name}")
                else:
                    self.logger.info(f"Retry attempt {attempt}/{max_retries} for operation: {operation_name}")
                    self.execution_stats["total_retries"] += 1

                # Execute the operation
                result = self.execute_operation(
                    operation=operation,
                    params=params,
                    progress_tracker=progress_tracker
                )

                # Add the attempt time to total execution time
                attempt_time = time.time() - attempt_start_time
                total_execution_time += attempt_time

                # If the operation was retried, update statistics
                if attempt > 0:
                    self.execution_stats["retried_operations"] += 1

                # Check the result status
                if result.status != OperationStatus.ERROR:
                    # Operation succeeded or completed with warnings
                    # Update the execution time to reflect total time across all attempts
                    result.execution_time = total_execution_time
                    return result

                # Operation failed with ERROR status, but we'll treat this as non-retriable
                # since it didn't raise an exception
                result.execution_time = total_execution_time
                return result

            except KeyboardInterrupt:
                # Always allow keyboard interrupts to propagate
                self.logger.info(f"Keyboard interrupt detected during operation: {operation_name}")
                raise

            except Exception as e:
                # Check for KeyboardInterrupt specifically before general exception handling
                if isinstance(e, KeyboardInterrupt):
                    self.logger.info(f"Keyboard interrupt detected during operation: {operation_name}")
                    raise

                # Add the attempt time to total execution time
                attempt_time = time.time() - attempt_start_time
                total_execution_time += attempt_time

                # Check if the exception is retriable
                if not self.is_retriable_error(e):
                    self.logger.error(
                        f"Operation {operation_name} failed with non-retriable exception: {str(e)}"
                    )
                    # Wrap in NonRetriableError to indicate it was explicitly non-retriable
                    raise NonRetriableError(str(e)) from e

                # Check if we've reached max retries
                if attempt >= max_retries:
                    self.logger.error(
                        f"Maximum retry attempts ({max_retries}) reached for operation {operation_name}"
                    )
                    # Raise MaxRetriesExceededError as promised in the docstring
                    raise MaxRetriesExceededError(
                        f"{str(e)} (Maximum retry attempts {max_retries} reached)"
                    ) from e

                # Calculate wait time for next retry
                wait_time = self._calculate_wait_time(
                    attempt + 1, backoff_factor, initial_wait, max_wait, jitter
                )

                # Call the retry callback if provided, using current exception
                if on_retry:
                    on_retry(e, attempt + 1, wait_time)

                self.logger.info(
                    f"Waiting {wait_time:.2f} seconds before retry attempt {attempt + 1}"
                )
                time.sleep(wait_time)

        # If we somehow reach here - which should be impossible due to the logic above,
        # but static type checkers require explicit return or exception handling
        raise ExecutionError(f"Retry logic exited unexpectedly for operation {operation_name}")

    def execute_operations(self,
                           operations: List[BaseOperation],
                           common_params: Dict[str, Any],
                           progress_tracker: Optional[ProgressTracker] = None,
                           continue_on_error: Optional[bool] = None) -> Dict[str, OperationResult]:
        """
        Execute a list of operations sequentially.

        Args:
            operations: List of operations to execute
            common_params: Common parameters for all operations
            progress_tracker: Progress tracker for all operations
            continue_on_error: Whether to continue executing operations after an error

        Returns:
            Dictionary mapping operation names to their results
        """
        # Use provided value or default from config
        if continue_on_error is None:
            continue_on_error = getattr(self.config, 'continue_on_error', False)

        results = {}

        # Create a subtask progress tracker if provided
        if progress_tracker:
            operations_progress = progress_tracker.create_subtask(
                len(operations), description="Executing operations"
            )
        else:
            operations_progress = None

        # Execute operations sequentially
        for i, operation in enumerate(operations):
            operation_name = operation.__class__.__name__
            operation_params = common_params.copy()

            # Create operation-specific progress tracker if main progress is provided
            if operations_progress:
                operation_progress = operations_progress.create_subtask(
                    100, description=f"Operation {i + 1}/{len(operations)}: {operation_name}"
                )
                operation_params['progress_tracker'] = operation_progress
            else:
                operation_progress = None

            # Add operation to reporter if available
            if self.reporter:
                self.reporter.add_operation(
                    name=f"Start {operation_name}",
                    status="running",
                    details={"index": i}
                )

            # Execute operation with retry
            try:
                result = self.execute_with_retry(
                    operation=operation,
                    params=operation_params,
                    progress_tracker=operation_progress
                )

                # Store the result
                results[operation_name] = result

                # Add operation result to reporter if available
                if self.reporter:
                    status_name = result.status.name.lower() if hasattr(result.status, 'name') else str(
                        result.status).lower()
                    self.reporter.add_operation(
                        name=f"Complete {operation_name}",
                        status=status_name,
                        details={
                            "execution_time": result.execution_time,
                            "metrics": result.metrics if hasattr(result, 'metrics') else {},
                            "error_message": result.error_message,
                            "error_trace": result.error_trace
                        }
                    )

                # Update progress tracker if provided
                if operations_progress:
                    operations_progress.update(i + 1)

                # Check result status
                if result.status == OperationStatus.ERROR and not continue_on_error:
                    self.logger.error(
                        f"Operation {operation_name} failed and continue_on_error is False. "
                        f"Aborting remaining operations."
                    )
                    break

            except KeyboardInterrupt:
                # Pass through KeyboardInterrupt to terminate execution
                self.logger.info("Operation execution interrupted by user (KeyboardInterrupt)")
                raise

            except (NonRetriableError, MaxRetriesExceededError, ExecutionError) as e:
                self.logger.exception(
                    f"Error executing operation {operation_name}: {str(e)}"
                )

                # Create error result with execution time
                execution_time = 0.0  # We don't have access to execution time here
                error_result = self._make_error_result(e, execution_time)

                # Store the result
                results[operation_name] = error_result

                # Add operation error to reporter if available
                if self.reporter:
                    self.reporter.add_operation(
                        name=f"Error {operation_name}",
                        status="error",
                        details={
                            "execution_time": error_result.execution_time,
                            "error_message": error_result.error_message,
                            "error_trace": error_result.error_trace
                        }
                    )

                # Update progress tracker if provided
                if operations_progress:
                    operations_progress.update(i + 1)

                # Check if we should continue
                if not continue_on_error:
                    self.logger.error(
                        f"Operation {operation_name} failed and continue_on_error is False. "
                        f"Aborting remaining operations."
                    )
                    break

            except Exception as e:
                # Handle other exceptions
                # Immediately re-raise KeyboardInterrupt to allow process termination
                if isinstance(e, KeyboardInterrupt):
                    raise

                self.logger.exception(
                    f"Unexpected exception in operation {operation_name}: {str(e)}"
                )

                # Create error result
                execution_time = 0.0  # We don't have access to execution time here
                error_result = self._make_error_result(e, execution_time)

                # Store the result
                results[operation_name] = error_result

                # Add operation error to reporter if available
                if self.reporter:
                    self.reporter.add_operation(
                        name=f"Error {operation_name}",
                        status="error",
                        details={
                            "execution_time": error_result.execution_time,
                            "error_message": error_result.error_message,
                            "error_trace": error_result.error_trace
                        }
                    )

                # Update progress tracker if provided
                if operations_progress:
                    operations_progress.update(i + 1)

                # Check if we should continue
                if not continue_on_error:
                    self.logger.error(
                        f"Unexpected exception in operation {operation_name} and continue_on_error is False. "
                        f"Aborting remaining operations."
                    )
                    break

        return results

    def execute_operations_parallel(self,
                                    operations: List[BaseOperation],
                                    common_params: Dict[str, Any],
                                    max_workers: Optional[int] = None,
                                    progress_tracker: Optional[ProgressTracker] = None,
                                    continue_on_error: Optional[bool] = None) -> Dict[str, OperationResult]:
        """
        Execute operations in parallel using multiple processes.

        Args:
            operations: List of operations to execute
            common_params: Common parameters for all operations
            max_workers: Maximum number of worker processes
            progress_tracker: Progress tracker for all operations
            continue_on_error: Whether to continue executing operations after an error

        Returns:
            Dictionary mapping operation names to their results

        Note:
            For parallel execution, all BaseOperation instances must be pickleable.
            Progress tracking in parallel mode has limitations due to process separation.
        """
        # Use provided value or default from config
        if continue_on_error is None:
            continue_on_error = getattr(self.config, 'continue_on_error', False)

        results = {}
        error_occurred = False

        # Determine max workers if not provided
        if max_workers is None:
            max_workers = getattr(self.config, 'parallel_processes', None)

        # Create operation dict with names
        named_operations = {op.__class__.__name__: op for op in operations}

        self.logger.info(f"Executing {len(operations)} operations in parallel with {max_workers} workers")

        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Create futures for each operation
                futures_to_names = {}

                for name, operation in named_operations.items():
                    # Copy parameters to avoid shared state issues
                    op_params = common_params.copy()

                    # Submit the task to the executor
                    future = executor.submit(
                        self.execute_with_retry,
                        operation=operation,
                        params=op_params
                    )
                    futures_to_names[future] = name

                # Process completed futures as they finish
                for future in as_completed(futures_to_names):
                    name = futures_to_names[future]

                    try:
                        # Get the result from the future
                        result = future.result()
                        results[name] = result

                        # Update statistics and log
                        if result.status == OperationStatus.SUCCESS:
                            self.logger.info(f"Parallel operation {name} completed successfully")
                        else:
                            self.logger.warning(
                                f"Parallel operation {name} completed with status: {result.status.name}")

                            # Check if we need to cancel remaining operations
                            if result.status == OperationStatus.ERROR and not continue_on_error and not error_occurred:
                                error_occurred = True
                                self.logger.error(
                                    f"Operation {name} failed and continue_on_error is False. "
                                    f"Canceling remaining operations."
                                )
                                # Cancel remaining futures
                                for f in futures_to_names.keys():
                                    if not f.done():
                                        f.cancel()

                                # Shutdown executor
                                executor.shutdown(wait=False)
                                break

                    except KeyboardInterrupt:
                        # Immediately cancel all tasks and re-raise to terminate
                        self.logger.info("Parallel execution interrupted by user (KeyboardInterrupt)")
                        for f in futures_to_names.keys():
                            if not f.done():
                                f.cancel()
                        executor.shutdown(wait=False)
                        raise

                    except Exception as e:
                        # Check for KeyboardInterrupt here as well
                        if isinstance(e, KeyboardInterrupt):
                            # Cancel all tasks and re-raise
                            for f in futures_to_names.keys():
                                if not f.done():
                                    f.cancel()
                            executor.shutdown(wait=False)
                            raise

                        self.logger.exception(f"Error in parallel operation {name}: {e}")

                        # Create error result
                        error_result = self._make_error_result(e, 0.0)
                        results[name] = error_result

                        # Check if we need to cancel remaining operations
                        if not continue_on_error and not error_occurred:
                            error_occurred = True
                            self.logger.error(
                                f"Operation {name} failed with exception and continue_on_error is False. "
                                f"Canceling remaining operations."
                            )
                            # Cancel remaining futures
                            for f in futures_to_names.keys():
                                if not f.done():
                                    f.cancel()

                            # Shutdown executor
                            executor.shutdown(wait=False)
                            break

            return results

        except KeyboardInterrupt:
            # Handle KeyboardInterrupt during executor setup/teardown
            self.logger.info("Parallel execution interrupted by user")
            raise

        except Exception as e:
            # Handle other exceptions during executor setup/teardown
            if isinstance(e, KeyboardInterrupt):
                self.logger.info("Parallel execution interrupted by user")
                raise

            self.logger.exception(f"Error setting up parallel execution: {e}")

            # Fall back to sequential execution
            self.logger.warning("Falling back to sequential execution due to parallel execution error")
            return self.execute_operations(
                operations=operations,
                common_params=common_params,
                progress_tracker=progress_tracker,
                continue_on_error=continue_on_error
            )

    def get_execution_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.

        Returns:
            Dictionary with execution statistics
        """
        return self.execution_stats.copy()

    def _calculate_wait_time(self,
                             attempt: int,
                             backoff_factor: float,
                             initial_wait: float,
                             max_wait: float,
                             jitter: bool) -> float:
        """
        Calculate wait time for the next retry attempt.

        Args:
            attempt: Current attempt number (1-based)
            backoff_factor: Factor for exponential backoff
            initial_wait: Initial wait time in seconds
            max_wait: Maximum wait time in seconds
            jitter: Whether to add jitter to wait times

        Returns:
            Wait time in seconds
        """
        # Calculate base wait time with exponential backoff
        wait = initial_wait * (backoff_factor ** (attempt - 1))

        # Apply maximum wait time constraint
        wait = min(wait, max_wait)

        # Add jitter if requested (up to 25% variation)
        if jitter:
            jitter_factor = 1.0 + random.uniform(-0.25, 0.25)
            wait *= jitter_factor

        return wait

    def _format_exception(self, exception: Exception) -> str:
        """
        Format an exception for logging and reporting.

        This method masks sensitive information in the traceback if configured.

        Args:
            exception: The exception to format

        Returns:
            Formatted exception string
        """
        import traceback

        # Get formatted traceback
        tb_str = ''.join(traceback.format_exception(
            type(exception), exception, exception.__traceback__
        ))

        # Check if we should mask sensitive information
        should_mask = getattr(self.config, 'mask_sensitive_data', False)
        if should_mask:
            # Get list of sensitive patterns to mask
            sensitive_patterns = getattr(self.config, 'sensitive_patterns', [
                r'password=\S+',
                r'key=\S+',
                r'token=\S+',
                r'secret=\S+',
                r'pwd=\S+'
            ])

            # Apply masking
            import re
            for pattern in sensitive_patterns:
                tb_str = re.sub(pattern, lambda m: m.group(0).split('=')[0] + '=****', tb_str)

        return tb_str


# Helper function to create an operation executor
def create_operation_executor(task_config: Any,
                              logger: logging.Logger,
                              reporter: Optional[TaskReporter] = None,
                              **kwargs) -> TaskOperationExecutor:
    """
    Create an operation executor for a task.

    Args:
        task_config: Task configuration object
        logger: Logger for operation execution
        reporter: Task reporter for tracking operation results
        **kwargs: Additional configuration parameters

    Returns:
        TaskOperationExecutor instance
    """
    return TaskOperationExecutor(
        task_config=task_config,
        logger=logger,
        reporter=reporter,
        **kwargs
    )