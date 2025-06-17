# TaskOperationExecutor Module Documentation

## Overview

The `operation_executor.py` module provides robust execution functionality for operations within the PAMOLA Core framework, focusing on resilience through retry capabilities, error handling, and progress tracking. It's designed to execute operations reliably even when faced with transient failures by implementing configurable retry strategies.

## Key Features

- **Resilient Operation Execution**: Execute operations with configurable retry strategies
- **Exponential Backoff**: Intelligent wait time calculation with exponential backoff and jitter
- **Selective Retry**: Smart filtering of which exceptions should trigger retry attempts
- **Progress Tracking**: Seamless integration with the progress tracking system
- **Parallel Execution**: Support for parallel operation execution using process pools
- **Comprehensive Logging**: Detailed logging of execution attempts and outcomes
- **Error Classification**: Intelligent classification of errors as retriable or non-retriable
- **Statistics Collection**: Tracking of execution metrics for reporting

## Dependencies

- `logging`, `random`, `time`: Standard library modules
- `concurrent.futures`: For parallel execution
- `pamola_core.utils.ops.op_base`: Base operation class
- `pamola_core.utils.ops.op_result`: Operation result class
- `pamola_core.utils.progress`: Progress tracking
- `pamola_core.utils.tasks.task_reporting`: Task reporting

## Exception Classes

- **ExecutionError**: Base exception for operation execution errors
- **MaxRetriesExceededError**: Exception raised when maximum retry attempts are reached
- **NonRetriableError**: Exception raised for errors that should not be retried

## Main Class

### TaskOperationExecutor

#### Description

Executor for task operations with retry capabilities. This class handles the execution of operations within a task, with support for retry logic, progress tracking, and result collection. It provides configurable retry strategies including exponential backoff and selective retry based on exception types.

#### Constructor

```python
def __init__(
    self,
    task_config: Any,
    logger: logging.Logger,
    reporter: Optional[TaskReporter] = None,
    default_max_retries: int = 3,
    default_backoff_factor: float = 2.0,
    default_initial_wait: float = 1.0,
    default_max_wait: float = 60.0,
    default_jitter: bool = True
)
```

**Parameters:**
- `task_config`: Task configuration object
- `logger`: Logger for operation execution
- `reporter`: Task reporter for tracking operation results (optional)
- `default_max_retries`: Default maximum retry attempts (default: 3)
- `default_backoff_factor`: Default backoff factor for exponential backoff (default: 2.0)
- `default_initial_wait`: Default initial wait time in seconds (default: 1.0)
- `default_max_wait`: Default maximum wait time in seconds (default: 60.0)
- `default_jitter`: Whether to add jitter to wait times (default: True)

#### Class Attributes

- **DEFAULT_RETRIABLE_EXCEPTIONS**: Set of exception types that are generally safe to retry
  - `ConnectionError`, `TimeoutError`, `IOError`, `BrokenPipeError`, etc.
- **NEVER_RETRY_EXCEPTIONS**: Set of exception types that should never be retried
  - `KeyboardInterrupt`, `SystemExit`, `MemoryError`, `NonRetriableError`, etc.

#### Key Methods

##### add_retriable_exception

```python
def add_retriable_exception(self, exception_type: ExceptionType) -> None
```

Add an exception type to the set of retriable exceptions.

**Parameters:**
- `exception_type`: The exception type to add

##### remove_retriable_exception

```python
def remove_retriable_exception(self, exception_type: ExceptionType) -> None
```

Remove an exception type from the set of retriable exceptions.

**Parameters:**
- `exception_type`: The exception type to remove

##### is_retriable_error

```python
def is_retriable_error(self, exception: Exception) -> bool
```

Determine if an exception should trigger a retry.

**Parameters:**
- `exception`: The exception to check

**Returns:**
- True if the exception is retriable, False otherwise

##### execute_operation

```python
def execute_operation(
    self,
    operation: BaseOperation,
    params: Dict[str, Any],
    progress_tracker: Optional[ProgressTracker] = None
) -> OperationResult
```

Execute a single operation without retry logic.

**Parameters:**
- `operation`: The operation to execute
- `params`: Parameters for the operation
- `progress_tracker`: Progress tracker for the operation (optional)

**Returns:**
- `OperationResult` containing the execution result

**Raises:**
- Exception: Any exception raised by the operation is propagated up

##### execute_with_retry

```python
def execute_with_retry(
    self,
    operation: BaseOperation,
    params: Dict[str, Any],
    max_retries: Optional[int] = None,
    backoff_factor: Optional[float] = None,
    initial_wait: Optional[float] = None,
    max_wait: Optional[float] = None,
    jitter: Optional[bool] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> OperationResult
```

Execute an operation with retry logic.

**Parameters:**
- `operation`: The operation to execute
- `params`: Parameters for the operation
- `max_retries`: Maximum number of retry attempts (optional)
- `backoff_factor`: Backoff factor for exponential backoff (optional)
- `initial_wait`: Initial wait time in seconds (optional)
- `max_wait`: Maximum wait time in seconds (optional)
- `jitter`: Whether to add jitter to wait times (optional)
- `progress_tracker`: Progress tracker for the operation (optional)
- `on_retry`: Callback function called before each retry attempt (optional)

**Returns:**
- `OperationResult` containing the execution result

**Raises:**
- `MaxRetriesExceededError`: If maximum retry attempts are reached
- `NonRetriableError`: If an exception is raised that should not be retried

**Note:**
The execution_time in the returned OperationResult represents the total time across all retry attempts, not just the final successful attempt.

##### execute_operations

```python
def execute_operations(
    self,
    operations: List[BaseOperation],
    common_params: Dict[str, Any],
    progress_tracker: Optional[ProgressTracker] = None,
    continue_on_error: Optional[bool] = None
) -> Dict[str, OperationResult]
```

Execute a list of operations sequentially.

**Parameters:**
- `operations`: List of operations to execute
- `common_params`: Common parameters for all operations
- `progress_tracker`: Progress tracker for all operations (optional)
- `continue_on_error`: Whether to continue executing operations after an error (optional)

**Returns:**
- Dictionary mapping operation names to their results

##### execute_operations_parallel

```python
def execute_operations_parallel(
    self,
    operations: List[BaseOperation],
    common_params: Dict[str, Any],
    max_workers: Optional[int] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    continue_on_error: Optional[bool] = None
) -> Dict[str, OperationResult]
```

Execute operations in parallel using multiple processes.

**Parameters:**
- `operations`: List of operations to execute
- `common_params`: Common parameters for all operations
- `max_workers`: Maximum number of worker processes (optional)
- `progress_tracker`: Progress tracker for all operations (optional)
- `continue_on_error`: Whether to continue executing operations after an error (optional)

**Returns:**
- Dictionary mapping operation names to their results

**Note:**
For parallel execution, all BaseOperation instances must be pickleable. Progress tracking in parallel mode has limitations due to process separation.

##### get_execution_stats

```python
def get_execution_stats(self) -> Dict[str, Any]
```

Get execution statistics.

**Returns:**
- Dictionary with execution statistics including:
  - `total_operations`: Number of operations executed
  - `successful_operations`: Number of operations executed successfully
  - `failed_operations`: Number of operations that failed
  - `retried_operations`: Number of operations that were retried
  - `total_retries`: Total number of retry attempts

#### Internal Methods

##### _calculate_wait_time

```python
def _calculate_wait_time(
    self,
    attempt: int,
    backoff_factor: float,
    initial_wait: float,
    max_wait: float,
    jitter: bool
) -> float
```

Calculate wait time for the next retry attempt.

**Parameters:**
- `attempt`: Current attempt number (1-based)
- `backoff_factor`: Factor for exponential backoff
- `initial_wait`: Initial wait time in seconds
- `max_wait`: Maximum wait time in seconds
- `jitter`: Whether to add jitter to wait times

**Returns:**
- Wait time in seconds

##### _format_exception

```python
def _format_exception(self, exception: Exception) -> str
```

Format an exception for logging and reporting.

**Parameters:**
- `exception`: The exception to format

**Returns:**
- Formatted exception string

##### _make_error_result

```python
def _make_error_result(
    self,
    exception: Exception,
    execution_time: float,
    additional_message: Optional[str] = None
) -> OperationResult
```

Create an error OperationResult from an exception.

**Parameters:**
- `exception`: The exception that caused the error
- `execution_time`: Execution time in seconds
- `additional_message`: Additional message to append to error_message (optional)

**Returns:**
- OperationResult with ERROR status and appropriate error information

## Helper Function

### create_operation_executor

```python
def create_operation_executor(
    task_config: Any,
    logger: logging.Logger,
    reporter: Optional[TaskReporter] = None,
    **kwargs
) -> TaskOperationExecutor
```

Create an operation executor for a task.

**Parameters:**
- `task_config`: Task configuration object
- `logger`: Logger for operation execution
- `reporter`: Task reporter for tracking operation results (optional)
- `**kwargs`: Additional configuration parameters

**Returns:**
- TaskOperationExecutor instance

## Usage Examples

### Basic Usage with Retry

```python
from pamola_core.utils.tasks.operation_executor import create_operation_executor
from pamola_core.utils.ops.op_base import BaseOperation
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create operation executor
executor = create_operation_executor(
    task_config=task_config,
    logger=logger,
    reporter=reporter,
    default_max_retries=3,
    default_backoff_factor=2.0
)

# Define operation parameters
operation_params = {
    "data_source": data_source,
    "task_dir": task_dir,
    "other_param": "value"
}

# Execute operation with retry
try:
    result = executor.execute_with_retry(
        operation=my_operation,
        params=operation_params
    )
    
    # Check result
    if result.status.is_success():
        print(f"Operation completed successfully: {result.metrics}")
    else:
        print(f"Operation completed with warnings: {result.error_message}")
        
except MaxRetriesExceededError as e:
    print(f"Operation failed after multiple retries: {e}")
except NonRetriableError as e:
    print(f"Operation failed with non-retriable error: {e}")
```

### Sequential Execution of Multiple Operations

```python
from pamola_core.utils.tasks.operation_executor import create_operation_executor
from pamola_core.utils.progress import ProgressTracker

# Create operation executor
executor = create_operation_executor(
    task_config=task_config,
    logger=logger,
    reporter=reporter
)

# Create progress tracker
progress = ProgressTracker(
    total=len(operations),
    description="Executing operations",
    unit="operations"
)

# Define common parameters
common_params = {
    "data_source": data_source,
    "task_dir": task_dir
}

# Execute operations sequentially
results = executor.execute_operations(
    operations=operations,
    common_params=common_params,
    progress_tracker=progress,
    continue_on_error=True
)

# Process results
for operation_name, result in results.items():
    if result.status.is_success():
        print(f"Operation {operation_name} succeeded with metrics: {result.metrics}")
    else:
        print(f"Operation {operation_name} failed: {result.error_message}")
```

### Parallel Execution with Custom Retry Settings

```python
from pamola_core.utils.tasks.operation_executor import create_operation_executor
import multiprocessing

# Determine optimal number of workers
cpu_count = multiprocessing.cpu_count()
max_workers = max(1, cpu_count - 1)  # Leave one CPU for system

# Create operation executor
executor = create_operation_executor(
    task_config=task_config,
    logger=logger,
    reporter=reporter
)

# Execute operations in parallel
results = executor.execute_operations_parallel(
    operations=operations,
    common_params=common_params,
    max_workers=max_workers,
    continue_on_error=True
)

# Get execution statistics
stats = executor.get_execution_stats()
print(f"Executed {stats['total_operations']} operations")
print(f"Successful: {stats['successful_operations']}")
print(f"Failed: {stats['failed_operations']}")
print(f"Required retries: {stats['retried_operations']} operations with {stats['total_retries']} total retries")
```

### Customizing Retriable Exceptions

```python
from pamola_core.utils.tasks.operation_executor import create_operation_executor

# Create operation executor
executor = create_operation_executor(
    task_config=task_config,
    logger=logger
)

# Add custom exception types to retry list
class TemporaryNetworkError(Exception):
    pass

class ResourceTemporarilyUnavailableError(Exception):
    retriable = True  # This will be checked by is_retriable_error
    
executor.add_retriable_exception(TemporaryNetworkError)

# Remove exception types that shouldn't be retried in your case
executor.remove_retriable_exception(IOError)

# Execute with custom retry behavior
result = executor.execute_with_retry(
    operation=my_operation,
    params=operation_params,
    max_retries=5,
    initial_wait=2.0,
    max_wait=120.0
)
```

### Using Retry Callback

```python
from pamola_core.utils.tasks.operation_executor import create_operation_executor

# Create operation executor
executor = create_operation_executor(
    task_config=task_config,
    logger=logger
)

# Define retry callback
def on_retry_callback(exception, attempt, wait_time):
    logger.warning(f"Retry attempt {attempt} after error: {str(exception)}")
    logger.warning(f"Waiting {wait_time:.2f} seconds before retry")
    
    # Notify external systems about the retry
    notify_monitoring_system(
        operation=my_operation.__class__.__name__,
        attempt=attempt,
        error=str(exception)
    )

# Execute with retry callback
result = executor.execute_with_retry(
    operation=my_operation,
    params=operation_params,
    on_retry=on_retry_callback
)
```

## Integration with BaseTask

The `operation_executor.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.operation_executor = create_operation_executor(
    task_config=self.config,
    logger=self.logger,
    reporter=self.reporter,
    progress_manager=self.progress_manager
)

# In BaseTask.execute()
try:
    # Prepare operation parameters
    operation_params = self._prepare_operation_parameters(operation)
    
    # Execute operation with retry
    result = self.operation_executor.execute_with_retry(
        operation=operation,
        params=operation_params
    )
    
    # Store the result
    self.results[operation_name] = result
    
    # Process artifacts and metrics
    if hasattr(result, 'artifacts') and result.artifacts:
        self.artifacts.extend(result.artifacts)
    
    if hasattr(result, 'metrics') and result.metrics:
        self.metrics[operation_name] = result.metrics
        
except (MaxRetriesExceededError, NonRetriableError) as e:
    self.logger.error(f"Operation {operation_name} failed: {e}")
    self.status = "operation_error"
```

## Retry Logic Workflow

The retry mechanism follows this workflow:

1. **Attempt Execution**: Execute the operation normally
2. **Exception Handling**: If an exception occurs, check if it's retriable
3. **Retry Decision**:
   - If the exception is not retriable, fail immediately with `NonRetriableError`
   - If max retries reached, fail with `MaxRetriesExceededError`
   - Otherwise, schedule a retry
4. **Backoff Calculation**: Calculate wait time with exponential backoff
5. **Notification**: Call on_retry callback if provided
6. **Wait Period**: Wait for the calculated time
7. **Retry Attempt**: Try again from step 1

## Exponential Backoff Algorithm

The wait time between retry attempts is calculated using an exponential backoff algorithm:

```
wait_time = initial_wait * (backoff_factor ^ (attempt - 1))
```

This is then capped at `max_wait` to prevent excessive wait times.

If jitter is enabled, a random variation (±25%) is applied to prevent thundering herd problems in distributed systems.

## Exception Classification

Exceptions are classified as retriable or non-retriable using these criteria:

1. **Never Retry List**: Exceptions in `NEVER_RETRY_EXCEPTIONS` are never retried
2. **Retriable List**: Exceptions in `retriable_exceptions` are always retried
3. **Retriable Attribute**: If the exception has a `retriable` attribute, its value is used
4. **Default Behavior**: Otherwise, the exception is considered non-retriable

## Best Practices

1. **Set Appropriate Retry Limits**: Set max_retries based on operation criticality and failure impact

2. **Use Exponential Backoff**: Always use exponential backoff for retries to avoid overloading resources

3. **Enable Jitter**: Keep jitter enabled to prevent synchronization of retry attempts

4. **Classify Exceptions**: Properly classify exceptions as retriable or non-retriable

5. **Monitor Retry Statistics**: Track and monitor retry statistics to identify frequent failures

6. **Set Reasonable Timeouts**: Ensure operations have reasonable timeouts to prevent long waits

7. **Use Progress Tracking**: Integrate with progress tracking for better user experience

8. **Consider Transactional Safety**: Ensure operations are idempotent if they might be retried

9. **Handle Non-Retriable Errors**: Provide clear error messages for non-retriable errors

10. **Balance Parallel Execution**: Choose an appropriate number of workers for parallel execution

## Limitations and Considerations

1. **Python Process Model**: Parallel execution uses Python's multiprocessing, which has overhead due to process creation

2. **Pickleable Requirements**: Operations and parameters must be pickleable for parallel execution

3. **Progress Tracking Limitations**: Progress tracking in parallel mode has limitations due to process separation

4. **Memory Usage**: Parallel execution increases memory usage due to process isolation

5. **State Isolation**: Processes do not share state, so operations should be designed accordingly

6. **Deterministic Behavior**: Operations should produce the same result when retried (idempotent)

7. **Resource Leaks**: Ensure operations properly clean up resources even when interrupted

## Common Use Cases

1. **Network Operations**: Retry operations that interact with external systems

2. **Database Transactions**: Retry database operations that may fail due to temporary issues

3. **File Operations**: Retry file operations that may fail due to temporary locks

4. **API Calls**: Retry API calls that may fail due to rate limiting or temporary outages

5. **Resource Allocation**: Retry operations that allocate resources that may be temporarily unavailable

## Error Handling Strategy

The module implements a sophisticated error handling strategy:

1. **Categorization**: Errors are categorized as retriable or non-retriable
2. **Isolation**: Execution errors are isolated from business logic errors
3. **Reporting**: Detailed error information is captured for debugging
4. **Recovery**: Retry logic provides automatic recovery from transient failures
5. **Fallback**: Sequential fallback is provided if parallel execution fails