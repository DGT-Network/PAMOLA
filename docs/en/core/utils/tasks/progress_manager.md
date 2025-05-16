# ProgressManager Module Documentation

## Overview

The `progress_manager.py` module provides centralized management of progress bars and logging for PAMOLA Core tasks. It ensures a clean, coordinated display of execution progress without conflicts between progress indicators and log messages. This module is essential for providing a smooth user experience during long-running privacy-enhancing operations, particularly when multiple operations are running in sequence with different progress stages.

## Key Features

- **Hierarchical Progress Bars**: Nests operation progress within task progress
- **Fixed Positioning**: Maintains progress bars at consistent console positions
- **Coordinated Logging**: Ensures logs don't break progress displays
- **Memory and Performance Metrics**: Tracks execution time and memory usage
- **Context Manager Support**: Simplifies operation progress tracking
- **Custom Metrics Collection**: Captures detailed progress statistics
- **Thread Safety**: Ensures consistent display in multi-threaded environments
- **Automatic Resource Cleanup**: Properly closes progress bars on completion
- **Quiet Mode Support**: Automatically detects non-interactive environments
- **Integration with Task Reporting**: Coordinates with the reporting system

## Dependencies

- `logging`, `sys`: Logging and standard I/O
- `threading`: Thread synchronization
- `time`: Time measurement
- `typing`: Type annotations
- `tqdm`: Progress bar implementation
- `psutil` (optional): Enhanced memory usage tracking

## Classes

### ProgressTrackerProtocol

#### Description

Protocol defining the interface for progress trackers. This ensures consistent API across different tracker implementations.

#### Required Methods

- `update(steps: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None`
- `set_description(description: str) -> None`
- `set_postfix(postfix: Dict[str, Any]) -> None`
- `close(failed: bool = False) -> None`

### NoOpProgressTracker

#### Description

No-operation progress tracker that implements the ProgressTrackerProtocol but doesn't display any output. Used in quiet mode or for testing.

#### Constructor

```python
def __init__(
    self,
    total: int,
    description: str,
    unit: str = "items",
    position: int = 0,
    leave: bool = True,
    parent: Optional[Any] = None,
    color: Optional[str] = None
)
```

**Parameters:**
- `total`: Total number of steps (unused)
- `description`: Description of the operation (unused)
- `unit`: Unit of progress (unused)
- `position`: Fixed position on screen (unused)
- `leave`: Whether to leave the progress bar after completion (unused)
- `parent`: Parent progress tracker (unused)
- `color`: Color of the progress bar (unused)

#### Methods

##### update

```python
def update(self, steps: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None
```

Updates metrics but doesn't display progress.

**Parameters:**
- `steps`: Number of steps completed
- `postfix`: Dictionary of metrics (unused visually but stored)

##### set_description

```python
def set_description(self, description: str) -> None
```

Updates the internal description without displaying.

**Parameters:**
- `description`: New description text

##### set_postfix

```python
def set_postfix(self, postfix: Dict[str, Any]) -> None
```

No-op version of set_postfix.

**Parameters:**
- `postfix`: Dictionary of metrics (unused)

##### close

```python
def close(self, failed: bool = False) -> None
```

Calculates final metrics without closing any visual display.

**Parameters:**
- `failed`: Whether the operation failed

### ProgressTracker

#### Description

Progress tracker for individual operations with fixed positioning. Wraps the tqdm library with additional functionality for hierarchical display, metrics collection, memory tracking, and proper positioning.

#### Constructor

```python
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
)
```

**Parameters:**
- `total`: Total number of steps
- `description`: Description of the operation
- `unit`: Unit of progress (e.g., "items", "records")
- `position`: Fixed position on screen (0 = top)
- `leave`: Whether to leave the progress bar after completion
- `parent`: Parent progress tracker (for hierarchical display)
- `color`: Color of the progress bar (None for default)
- `disable`: Whether to disable the progress bar display

#### Key Attributes

- `total`: Total number of steps
- `description`: Description of the operation
- `unit`: Unit of progress
- `position`: Fixed position on screen
- `leave`: Whether to leave the progress bar after completion
- `parent`: Parent progress tracker
- `color`: Color of the progress bar
- `children`: List of child progress trackers
- `disabled`: Whether the progress bar is disabled
- `start_time`: Time when the tracker was created
- `start_memory`: Memory usage when the tracker was created
- `peak_memory`: Maximum memory usage during operation
- `metrics`: Dictionary of custom metrics
- `pbar`: tqdm progress bar instance

#### Methods

##### update

```python
def update(self, steps: int = 1, postfix: Optional[Dict[str, Any]] = None) -> None
```

Updates progress by specified number of steps.

**Parameters:**
- `steps`: Number of steps completed
- `postfix`: Dictionary of metrics to display after the progress bar

##### set_description

```python
def set_description(self, description: str) -> None
```

Updates the description of the progress bar.

**Parameters:**
- `description`: New description text

##### set_postfix

```python
def set_postfix(self, postfix: Dict[str, Any]) -> None
```

Sets the postfix metrics display.

**Parameters:**
- `postfix`: Dictionary of metrics to display

##### close

```python
def close(self, failed: bool = False) -> None
```

Closes the progress bar and computes final metrics.

**Parameters:**
- `failed`: Whether the operation failed

##### clear

```python
def clear(self) -> None
```

Temporarily clears the progress bar from display.

##### refresh

```python
def refresh(self) -> None
```

Redraws the progress bar after clearing.

##### _get_current_memory

```python
def _get_current_memory(self) -> float
```

Gets current memory usage in MB.

**Returns:**
- Memory usage in MB

### TaskProgressManager

#### Description

Centralized manager for task progress and logging coordination. Coordinates progress display and logging to ensure they don't interfere with each other, creating a clean user experience.

#### Constructor

```python
def __init__(
    self,
    task_id: str,
    task_type: str,
    logger: logging.Logger,
    reporter: Optional[Any] = None,
    total_operations: int = 0,
    quiet: bool = False
)
```

**Parameters:**
- `task_id`: Task identifier
- `task_type`: Type of task
- `logger`: Logger for the task
- `reporter`: Task reporter for metrics
- `total_operations`: Total number of operations (if known)
- `quiet`: Whether to disable progress bars

#### Key Attributes

- `task_id`: Task identifier
- `task_type`: Type of task
- `logger`: Logger for the task
- `reporter`: Task reporter for metrics
- `total_operations`: Total number of operations
- `quiet`: Whether progress bars are disabled
- `lock`: Thread lock for synchronization
- `operations_completed`: Number of completed operations
- `active_operations`: Dictionary of active progress trackers
- `start_time`: Time when the manager was created
- `peak_memory`: Maximum memory usage during task execution
- `main_progress`: Main progress bar for overall task progress

#### Methods

##### _check_logger_handlers

```python
def _check_logger_handlers(self) -> None
```

Checks logger handlers to prevent conflicts with progress bars.

##### start_operation

```python
def start_operation(
    self,
    name: str,
    total: int,
    description: Optional[str] = None,
    unit: str = "items",
    leave: bool = False
) -> Union[ProgressTracker, NoOpProgressTracker]
```

Starts tracking a new operation.

**Parameters:**
- `name`: Operation name (unique identifier)
- `total`: Total number of steps in the operation
- `description`: Description of the operation (defaults to name)
- `unit`: Unit of progress
- `leave`: Whether to leave the progress bar after completion

**Returns:**
- Progress tracker for the operation

##### update_operation

```python
def update_operation(
    self,
    name: str,
    steps: int = 1,
    postfix: Optional[Dict[str, Any]] = None
) -> None
```

Updates progress of a specific operation.

**Parameters:**
- `name`: Operation name
- `steps`: Number of steps completed
- `postfix`: Additional metrics to display

##### complete_operation

```python
def complete_operation(
    self,
    name: str,
    success: bool = True,
    metrics: Optional[Dict[str, Any]] = None
) -> None
```

Marks an operation as completed.

**Parameters:**
- `name`: Operation name
- `success`: Whether the operation completed successfully
- `metrics`: Final metrics for the operation

##### log_message

```python
def log_message(
    self,
    level: int,
    message: str,
    preserve_progress: bool = True
) -> None
```

Logs a message without breaking progress bars.

**Parameters:**
- `level`: Logging level
- `message`: Message to log
- `preserve_progress`: Whether to preserve progress bars after logging

##### Convenience Logging Methods

```python
def log_info(self, message: str) -> None
def log_warning(self, message: str) -> None
def log_error(self, message: str) -> None
def log_debug(self, message: str) -> None
def log_critical(self, message: str, preserve_progress: bool = False) -> None
```

Convenience methods for logging at different levels.

##### create_operation_context

```python
def create_operation_context(
    self,
    name: str,
    total: int,
    description: Optional[str] = None,
    unit: str = "items",
    leave: bool = False
) -> 'ProgressContext'
```

Creates a context manager for an operation.

**Parameters:**
- `name`: Operation name
- `total`: Total number of steps
- `description`: Description of the operation
- `unit`: Unit of progress
- `leave`: Whether to leave the progress bar after completion

**Returns:**
- Context manager for the operation

##### get_metrics

```python
def get_metrics(self) -> Dict[str, Any]
```

Gets overall metrics for the task.

**Returns:**
- Dictionary of task metrics

##### close

```python
def close(self) -> None
```

Closes all progress bars and releases resources.

##### _get_current_memory and _get_initial_memory

```python
def _get_current_memory(self) -> float
def _get_initial_memory(self) -> float
```

Helper methods for memory usage tracking.

### ProgressContext

#### Description

Context manager for operation execution with progress tracking. Provides a convenient way to track progress of an operation using the context manager pattern (with statement).

#### Constructor

```python
def __init__(
    self,
    progress_manager: TaskProgressManager,
    operation_name: str,
    total: int,
    description: Optional[str] = None,
    unit: str = "items",
    leave: bool = False
)
```

**Parameters:**
- `progress_manager`: Task progress manager
- `operation_name`: Name of the operation
- `total`: Total number of steps
- `description`: Description of the operation
- `unit`: Unit of progress
- `leave`: Whether to leave the progress bar after completion

#### Methods

##### __enter__

```python
def __enter__(self) -> Union[ProgressTracker, NoOpProgressTracker]
```

Starts tracking operation progress.

**Returns:**
- Progress tracker for the operation

##### __exit__

```python
def __exit__(
    self,
    exc_type,
    exc_val,
    exc_tb
) -> None
```

Completes operation tracking.

**Parameters:**
- `exc_type`: Exception type if an exception occurred
- `exc_val`: Exception value if an exception occurred
- `exc_tb`: Exception traceback if an exception occurred

## Utility Functions

### create_task_progress_manager

```python
def create_task_progress_manager(
    task_id: str,
    task_type: str,
    logger: logging.Logger,
    reporter: Optional[Any] = None,
    total_operations: int = 0,
    quiet: Optional[bool] = None
) -> TaskProgressManager
```

Creates a task progress manager with auto-detection of quiet mode.

**Parameters:**
- `task_id`: Task identifier
- `task_type`: Type of task
- `logger`: Logger for the task
- `reporter`: Task reporter for metrics
- `total_operations`: Total number of operations (if known)
- `quiet`: Whether to disable progress bars (auto-detected if None)

**Returns:**
- Task progress manager

## Usage Examples

### Basic Progress Tracking

```python
import logging
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1P",
    task_type="profiling",
    logger=logger,
    total_operations=3
)

# Start a new operation
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    # Simulate work with progress updates
    for i in range(10):
        # Do work
        import time
        time.sleep(0.1)
        
        # Update progress
        progress.update(10, {"stage": f"chunk {i+1}/10"})
        
        # Log info without breaking progress bar
        if i == 5:
            progress_manager.log_info("Halfway through loading data")

# Operation is automatically completed when context exits

# Start another operation
with progress_manager.create_operation_context(
    name="analyze_data",
    total=50,
    description="Analyzing data"
) as progress:
    # Simulate work
    for i in range(5):
        time.sleep(0.2)
        progress.update(10)

# Manually start an operation
progress = progress_manager.start_operation(
    name="generate_report",
    total=10,
    description="Generating report"
)

# Update manually
progress.update(5, {"status": "formatting"})
progress.update(5, {"status": "complete"})

# Complete the operation
progress_manager.complete_operation(
    name="generate_report",
    success=True,
    metrics={"report_size_kb": 125}
)

# Get overall metrics
metrics = progress_manager.get_metrics()
print(f"Task metrics: {metrics}")

# Clean up
progress_manager.close()
```

### Hierarchical Progress Bars

```python
import logging
import time
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A",
    task_type="anonymization",
    logger=logger,
    total_operations=2
)

# Start main operation
with progress_manager.create_operation_context(
    name="anonymize_data",
    total=3,
    description="Anonymizing data"
) as main_progress:
    # First sub-operation
    with progress_manager.create_operation_context(
        name="quasi_identifier_processing",
        total=100,
        description="Processing quasi-identifiers"
    ) as sub_progress:
        for i in range(10):
            time.sleep(0.1)
            sub_progress.update(10)
    
    # Update main progress (happens automatically with context)
    main_progress.set_description("Anonymizing data (1/3 complete)")
    
    # Second sub-operation
    with progress_manager.create_operation_context(
        name="k_anonymity",
        total=100,
        description="Applying k-anonymity"
    ) as sub_progress:
        for i in range(10):
            time.sleep(0.1)
            sub_progress.update(10, {"k": 5})
    
    # Update main progress
    main_progress.set_description("Anonymizing data (2/3 complete)")
    
    # Third sub-operation
    with progress_manager.create_operation_context(
        name="validation",
        total=100,
        description="Validating results"
    ) as sub_progress:
        for i in range(10):
            time.sleep(0.1)
            sub_progress.update(10)
    
    # Main operation progress is updated automatically

# Start second operation
with progress_manager.create_operation_context(
    name="generate_report",
    total=100,
    description="Generating report"
) as progress:
    for i in range(10):
        time.sleep(0.1)
        progress.update(10)

# Close all progress bars
progress_manager.close()
```

### Integration with Error Handling

```python
import logging
import time
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1P",
    task_type="profiling",
    logger=logger,
    total_operations=2
)

# First operation - successful
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    for i in range(10):
        time.sleep(0.1)
        progress.update(10)

# Second operation - with error handling
try:
    with progress_manager.create_operation_context(
        name="analyze_data",
        total=100,
        description="Analyzing data"
    ) as progress:
        for i in range(10):
            time.sleep(0.1)
            progress.update(10)
            
            # Simulate an error
            if i == 5:
                raise ValueError("Simulated error during analysis")
except Exception as e:
    # Error is already logged by the context manager
    # operation is marked as failed automatically
    pass

# Get metrics to see operation status
metrics = progress_manager.get_metrics()
print(f"Operations completed: {metrics['operations_completed']}")

# Clean up
progress_manager.close()
```

### Integration with TaskReporter

```python
import logging
import time
from pathlib import Path
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager
from pamola_core.utils.tasks.task_reporting import TaskReporter

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A",
    task_type="anonymization",
    logger=logger,
    total_operations=2
)

# Create reporter with progress manager
reporter = TaskReporter(
    task_id="t_1A",
    task_type="anonymization",
    description="Anonymize customer data",
    report_path=Path("DATA/reports/t_1A_report.json"),
    progress_manager=progress_manager
)

# First operation
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    for i in range(10):
        time.sleep(0.1)
        progress.update(10)
    
    # Add operation to report
    reporter.add_operation(
        name="Load Data",
        status="success",
        details={
            "rows": 1000,
            "columns": 15,
            "execution_time": 1.0
        }
    )

# Second operation
with progress_manager.create_operation_context(
    name="anonymize_data",
    total=100,
    description="Anonymizing data"
) as progress:
    for i in range(10):
        time.sleep(0.1)
        progress.update(10)
    
    # Add operation to report
    reporter.add_operation(
        name="Anonymize Data",
        status="success",
        details={
            "method": "k-anonymity",
            "k": 5,
            "execution_time": 1.0
        }
    )

# Add artifact to report
reporter.add_artifact(
    artifact_type="csv",
    path="DATA/processed/t_1A/output/anonymized_data.csv",
    description="Anonymized dataset"
)

# Add task summary
reporter.add_task_summary(
    success=True,
    execution_time=progress_manager.get_metrics()["execution_time"]
)

# Save report
report_path = reporter.save_report()
print(f"Report saved to: {report_path}")

# Clean up
progress_manager.close()
```

### Using in Quiet Mode

```python
import logging
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager in quiet mode
progress_manager = create_task_progress_manager(
    task_id="t_1P",
    task_type="profiling",
    logger=logger,
    total_operations=2,
    quiet=True
)

# Operations work the same way, but no visual output
with progress_manager.create_operation_context(
    name="load_data",
    total=100,
    description="Loading data"
) as progress:
    # Do work
    import time
    for i in range(10):
        time.sleep(0.1)
        progress.update(10)
    
    # Logs still work in quiet mode
    progress_manager.log_info("Data loaded successfully")

# Get metrics (still collected in quiet mode)
metrics = progress_manager.get_metrics()
print(f"Operations completed: {metrics['operations_completed']}")
print(f"Execution time: {metrics['execution_time']:.2f} seconds")
print(f"Peak memory: {metrics['peak_memory_mb']:.2f} MB")

# Clean up
progress_manager.close()
```

## Integration with BaseTask

The `progress_manager.py` module is designed to integrate seamlessly with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.progress_manager = create_task_progress_manager(
    task_id=self.task_id,
    task_type=self.task_type,
    logger=self.logger,
    reporter=self.reporter,
    total_operations=0  # Will be updated after configure_operations()
)

# After configure_operations()
if self.operations:
    self.progress_manager.total_operations = len(self.operations)

# In execute() method
for i, operation in enumerate(self.operations):
    operation_name = operation.name if hasattr(operation, 'name') else f"Operation {i + 1}"
    
    with self.progress_manager.create_operation_context(
        name=operation_name,
        total=100,
        description=f"Executing: {operation_name}"
    ) as progress:
        # Execute operation with progress tracking
        operation_params["progress_tracker"] = progress
        result = operation.run(**operation_params)
        
        # Add metrics from operation result
        if hasattr(result, 'metrics') and result.metrics:
            for key, value in result.metrics.items():
                progress.metrics[key] = value
```

## Best Practices

1. **Use Context Managers**: Prefer the `create_operation_context()` method with `with` statements for cleaner code and automatic error handling.

2. **Hierarchical Structure**: Take advantage of the hierarchical structure to show both overall task progress and detailed operation progress.

3. **Meaningful Descriptions**: Use clear, informative descriptions that indicate both the operation and its current status.

4. **Update Regularly**: Update progress frequently enough to provide meaningful feedback (usually 5-10% increments).

5. **Include Status in Postfix**: Use the postfix parameter to show the current stage or status of the operation.

6. **Coordinate Logging**: Always use the progress manager's logging methods instead of direct logger calls to maintain clean display.

7. **Handle Errors Properly**: Let the context manager handle errors automatically, but provide clear error messages.

8. **Clean Up Resources**: Always call `close()` at the end of the task or use the progress manager as a context manager.

9. **Disable in CI**: Automatically disable progress bars in non-interactive environments by using the auto-detection in `create_task_progress_manager()`.

10. **Integrate with Reporting**: Use the progress manager together with TaskReporter for coordinated execution tracking.

## Console Output Examples

### Normal Mode Output

With hierarchical progress bars, the console output looks like:

```
Task: t_1A (anonymization): 50%|██████████                    | 1/2 [00:01<00:01, 1.00ops/s]
Operation: Loading data: 100%|████████████████████| 100/100 [00:01<00:00, 95.24items/s, mem=45.3MB]
Operation: Anonymizing data: 60%|████████████      | 60/100 [00:00<00:00, 95.40items/s, k=5]
[INFO] Processing quasi-identifiers complete
```

### Quiet Mode Output

In quiet mode, only log messages are shown:

```
[INFO] Starting operation: load_data
[INFO] Data loaded successfully
[INFO] Starting operation: anonymize_data
[INFO] Processing quasi-identifiers complete
[INFO] Operation load_data completed successfully
[INFO] Operation anonymize_data completed successfully
```

## Technical Details

### Thread Safety

The `TaskProgressManager` uses a threading lock to ensure thread-safe operations on shared progress bars, which is important for concurrent operations or background logging.

### Memory Tracking

Memory tracking is provided through two mechanisms:
1. Optional `psutil` library for accurate system-level memory tracking
2. Fallback mechanism that still provides timing metrics when `psutil` is not available

### Terminal Compatibility

The progress manager automatically detects terminal capabilities:
1. Disables progress bars in non-interactive environments (pipes, redirects)
2. Avoids breaking ANSI codes in CI environments
3. Falls back to simple logging in environments that don't support progress bars

### Performance Considerations

- Progress updates are relatively cheap but should still be batched when processing very large datasets
- Memory tracking adds minimal overhead when using `psutil`
- Locking for thread safety adds minimal overhead in single-threaded environments