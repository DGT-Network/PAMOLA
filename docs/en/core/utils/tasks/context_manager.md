# TaskContextManager Module Documentation

## Overview

The `context_manager.py` module provides functionality for managing task execution state in the PAMOLA Core framework. It enables checkpoint creation, state persistence, and resumable execution, allowing tasks to recover from interruptions or failures. This module is essential for building resilient task workflows that can tolerate interruptions and continue from where they left off.

## Key Features

- **Task State Serialization**: Safely saves and restores execution state
- **Automatic Checkpointing**: Creates checkpoints between operations
- **Resumable Execution**: Enables tasks to continue from last checkpoint
- **Execution State Verification**: Validates state consistency before restoration
- **Cross-Run Persistence**: Integrates with execution log for checkpoints that persist across runs
- **Atomic File Operations**: Ensures checkpoint files aren't corrupted during write operations
- **Checkpoint Management**: Automatically prunes old checkpoints to manage disk space
- **Progress Tracking Integration**: Provides visual feedback during checkpoint operations
- **Error Recovery**: Creates special error checkpoints when exceptions occur

## Dependencies

- `json`, `logging`, `os`, `tempfile`: Standard library modules
- `datetime`: Timestamp management
- `pathlib.Path`: Path manipulation
- `filelock`: Thread-safe file locking
- `pamola_core.utils.io`: File I/O utilities
- `pamola_core.utils.tasks.execution_log`: Execution log integration
- `pamola_core.utils.tasks.path_security`: Path security validation

## Constants

- `DEFAULT_LOCK_TIMEOUT`: Default timeout for file locks (10 seconds)
- `DEFAULT_MAX_CHECKPOINTS`: Default maximum number of checkpoints to keep (10)
- `DEFAULT_MAX_STATE_SIZE`: Default maximum state size before pruning (10 MB)

## Exception Classes

- **ContextManagerError**: Base exception for context manager errors
- **CheckpointError**: Exception raised for checkpoint-related errors
- **StateSerializationError**: Exception raised when state serialization fails
- **StateRestorationError**: Exception raised when state restoration fails

## Helper Classes

### NullProgressTracker

A no-op progress tracker that implements the required interface. Used when a real progress tracker is unavailable to prevent null reference errors.

```python
class NullProgressTracker:
    def update(self, steps=1, postfix=None): ...
    def set_description(self, description): ...
    def set_postfix(self, postfix): ...
    def close(self, failed=False): ...
```

## Main Class

### TaskContextManager

#### Description

Manager for task execution state with checkpoint support. This class provides functionality for managing task execution state, creating checkpoints, and enabling resumable execution after interruptions. It integrates with the execution log for consistent checkpoint tracking across task runs.

#### Constructor

```python
def __init__(
    self,
    task_id: str,
    task_dir: Path,
    max_state_size: int = DEFAULT_MAX_STATE_SIZE,
    progress_manager: Optional[Any] = None
)
```

**Parameters:**
- `task_id`: ID of the task
- `task_dir`: Path to the task directory
- `max_state_size`: Maximum state size in bytes before pruning older checkpoints
- `progress_manager`: Optional progress manager for visual feedback during operations

**Raises:**
- `ContextManagerError`: If initialization fails

#### Key Attributes

- `task_id`: ID of the task
- `task_dir`: Path to the task directory
- `max_state_size`: Maximum state size in bytes before pruning
- `progress_manager`: Progress manager for tracking
- `logger`: Logger for the context manager
- `checkpoint_dir`: Directory for storing checkpoints
- `lock_file`: File lock path for thread safety
- `current_state`: Dictionary containing the current execution state
- `checkpoints`: List of available checkpoints with timestamps

#### Key Methods

##### save_execution_state

```python
def save_execution_state(
    self, 
    state: Dict[str, Any], 
    checkpoint_name: Optional[str] = None
) -> Path
```

Save execution state to a checkpoint file.

**Parameters:**
- `state`: Execution state to save
- `checkpoint_name`: Name for the checkpoint (optional)

**Returns:**
- Path to the saved checkpoint file

**Raises:**
- `StateSerializationError`: If saving the state fails

##### restore_execution_state

```python
def restore_execution_state(
    self, 
    checkpoint_name: Optional[str] = None
) -> Dict[str, Any]
```

Restore execution state from a checkpoint file.

**Parameters:**
- `checkpoint_name`: Name of the checkpoint to restore (optional, uses latest if None)

**Returns:**
- Restored execution state dictionary

**Raises:**
- `StateRestorationError`: If restoring the state fails

##### create_automatic_checkpoint

```python
def create_automatic_checkpoint(
    self, 
    operation_index: int, 
    metrics: Dict[str, Any]
) -> str
```

Create an automatic checkpoint at the current execution point.

**Parameters:**
- `operation_index`: Index of the current operation
- `metrics`: Metrics collected up to this point

**Returns:**
- Name of the created checkpoint

**Raises:**
- `CheckpointError`: If creating the checkpoint fails

##### update_state

```python
def update_state(
    self, 
    updates: Dict[str, Any]
) -> None
```

Update the current execution state.

**Parameters:**
- `updates`: Dictionary of updates to apply to the current state

**Raises:**
- `ContextManagerError`: If updating the state fails

##### record_operation_completion

```python
def record_operation_completion(
    self, 
    operation_index: int, 
    operation_name: str,
    result_metrics: Optional[Dict[str, Any]] = None
) -> None
```

Record the completion of an operation.

**Parameters:**
- `operation_index`: Index of the completed operation
- `operation_name`: Name of the completed operation
- `result_metrics`: Metrics from the operation result

**Raises:**
- `ContextManagerError`: If recording the operation fails

##### record_operation_failure

```python
def record_operation_failure(
    self, 
    operation_index: int, 
    operation_name: str,
    error_info: Dict[str, Any]
) -> None
```

Record the failure of an operation.

**Parameters:**
- `operation_index`: Index of the failed operation
- `operation_name`: Name of the failed operation
- `error_info`: Information about the error

**Raises:**
- `ContextManagerError`: If recording the failure fails

##### record_artifact

```python
def record_artifact(
    self, 
    artifact_path: Union[str, Path], 
    artifact_type: str,
    description: Optional[str] = None
) -> None
```

Record an artifact created during task execution.

**Parameters:**
- `artifact_path`: Path to the artifact
- `artifact_type`: Type of the artifact
- `description`: Description of the artifact

**Raises:**
- `ContextManagerError`: If recording the artifact fails

##### can_resume_execution

```python
def can_resume_execution(self) -> Tuple[bool, Optional[str]]
```

Check if task execution can be resumed from a checkpoint.

**Returns:**
- Tuple containing:
  - Boolean indicating if execution can be resumed
  - Name of the checkpoint to resume from (or None)

**Raises:**
- `ContextManagerError`: If checking for resumable execution fails

##### get_latest_checkpoint

```python
def get_latest_checkpoint(self) -> Optional[str]
```

Get the name of the latest checkpoint from the execution log.

**Returns:**
- Name of the latest checkpoint or None if not found

##### get_current_state

```python
def get_current_state(self) -> Dict[str, Any]
```

Get the current execution state.

**Returns:**
- Current execution state dictionary

##### get_checkpoints

```python
def get_checkpoints(self) -> List[Tuple[str, datetime]]
```

Get a list of available checkpoints with timestamps.

**Returns:**
- List of tuples containing checkpoint names and timestamps, sorted by creation time (newest first)

##### cleanup_old_checkpoints

```python
def cleanup_old_checkpoints(
    self, 
    max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS
) -> int
```

Remove old checkpoints to manage disk space.

**Parameters:**
- `max_checkpoints`: Maximum number of checkpoints to keep

**Returns:**
- Number of checkpoints removed

**Raises:**
- `ContextManagerError`: If cleaning up checkpoints fails

##### cleanup

```python
def cleanup(self) -> None
```

Explicitly clean up resources.

This method is called when the manager is no longer needed, especially in cases where the context manager cannot be used.

#### Context Manager Support

The class implements the context manager protocol (`__enter__` and `__exit__`), allowing it to be used with the `with` statement:

```python
with TaskContextManager(task_id, task_dir) as context_manager:
    # Use context manager here
    # If an exception occurs, a final error checkpoint will be created
```

#### Internal Methods

##### _load_checkpoint_history

```python
def _load_checkpoint_history(self) -> None
```

Load existing checkpoint history.

##### _atomic_write_json

```python
def _atomic_write_json(
    self, 
    data: Dict[str, Any], 
    path: Path
) -> None
```

Write JSON data to a file atomically.

**Parameters:**
- `data`: JSON data to write
- `path`: Target file path

**Raises:**
- `StateSerializationError`: If writing fails

##### _sanitize_checkpoint_name

```python
def _sanitize_checkpoint_name(self, name: str) -> str
```

Sanitize a checkpoint name to ensure it's safe to use as a filename.

**Parameters:**
- `name`: Checkpoint name to sanitize

**Returns:**
- Sanitized checkpoint name

## Helper Function

### create_task_context_manager

```python
def create_task_context_manager(
    task_id: str,
    task_dir: Path,
    max_state_size: int = DEFAULT_MAX_STATE_SIZE,
    progress_manager: Optional[Any] = None
) -> TaskContextManager
```

Create a context manager for a task.

**Parameters:**
- `task_id`: ID of the task
- `task_dir`: Path to the task directory
- `max_state_size`: Maximum state size in bytes before pruning older checkpoints
- `progress_manager`: Optional progress manager for visual feedback

**Returns:**
- TaskContextManager instance

**Raises:**
- `ContextManagerError`: If context manager creation fails

## Execution State Structure

The execution state maintained by the context manager includes:

```json
{
  "task_id": "t_1A_profiling",
  "created_at": "2025-05-01T12:00:00",
  "last_updated": "2025-05-01T12:30:45",
  "operation_index": 2,
  "operations_completed": [
    {
      "index": 0,
      "name": "load_data",
      "timestamp": "2025-05-01T12:05:30"
    },
    {
      "index": 1,
      "name": "analyze_data",
      "timestamp": "2025-05-01T12:15:20"
    },
    {
      "index": 2,
      "name": "generate_report",
      "timestamp": "2025-05-01T12:30:45"
    }
  ],
  "operations_failed": [],
  "metrics": {
    "load_data": {
      "rows_processed": 10000,
      "execution_time": 5.3
    },
    "analyze_data": {
      "fields_analyzed": 15,
      "execution_time": 10.2
    },
    "generate_report": {
      "charts_generated": 5,
      "execution_time": 15.4
    }
  },
  "artifacts": [
    {
      "path": "DATA/processed/t_1A_profiling/output/field_statistics.json",
      "type": "json",
      "description": "Field statistics",
      "timestamp": "2025-05-01T12:20:30"
    },
    {
      "path": "DATA/processed/t_1A_profiling/output/profile_report.html",
      "type": "html",
      "description": "Profile report",
      "timestamp": "2025-05-01T12:30:45"
    }
  ]
}
```

## Usage Examples

### Basic Checkpoint Creation and Restoration

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pathlib import Path

# Create context manager
context_manager = create_task_context_manager(
    task_id="t_1A_profiling", 
    task_dir=Path("DATA/processed/t_1A_profiling")
)

# Manually create a checkpoint
state = {
    "task_id": "t_1A_profiling",
    "operation_index": 1,
    "operations_completed": [
        {"index": 0, "name": "load_data", "timestamp": "2025-05-01T12:05:30"}
    ],
    "metrics": {
        "load_data": {"rows_processed": 10000}
    }
}

# Save state to a checkpoint
checkpoint_path = context_manager.save_execution_state(state, "after_load_data")
print(f"Checkpoint saved to: {checkpoint_path}")

# Later, restore from the checkpoint
restored_state = context_manager.restore_execution_state("after_load_data")
print(f"Restored state with operation_index: {restored_state['operation_index']}")
```

### Automatic Checkpointing During Task Execution

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pathlib import Path

# Create context manager
context_manager = create_task_context_manager(
    task_id="t_1A_profiling", 
    task_dir=Path("DATA/processed/t_1A_profiling")
)

# Execute operations with checkpoints
try:
    # First operation
    print("Executing operation 1: load_data")
    # ...execution code...
    
    # Create checkpoint after operation 1
    context_manager.create_automatic_checkpoint(
        operation_index=0, 
        metrics={"rows_processed": 10000, "execution_time": 5.3}
    )
    
    # Second operation
    print("Executing operation 2: analyze_data")
    # ...execution code...
    
    # Record completion
    context_manager.record_operation_completion(
        operation_index=1, 
        operation_name="analyze_data",
        result_metrics={"fields_analyzed": 15, "execution_time": 10.2}
    )
    
    # Create another checkpoint
    context_manager.create_automatic_checkpoint(
        operation_index=1, 
        metrics=context_manager.get_current_state().get("metrics", {})
    )
    
except Exception as e:
    print(f"Task execution failed: {e}")
    # The context manager will create an error checkpoint automatically
```

### Resumable Task Execution

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pathlib import Path

# Create context manager
context_manager = create_task_context_manager(
    task_id="t_1A_profiling", 
    task_dir=Path("DATA/processed/t_1A_profiling")
)

# Check if task can be resumed
can_resume, checkpoint_name = context_manager.can_resume_execution()

if can_resume:
    print(f"Resuming execution from checkpoint: {checkpoint_name}")
    
    # Restore state from checkpoint
    state = context_manager.restore_execution_state(checkpoint_name)
    
    # Determine what operations have been completed
    completed_operations = [op["index"] for op in state.get("operations_completed", [])]
    last_operation_index = state.get("operation_index", -1)
    
    # Skip completed operations
    if 0 in completed_operations:
        print("Skipping operation 1: already completed")
    else:
        print("Executing operation 1")
        # ...execution code...
    
    if 1 in completed_operations:
        print("Skipping operation 2: already completed")
    else:
        print("Executing operation 2")
        # ...execution code...
    
    # Continue with remaining operations
    print("Executing operation 3")
    # ...execution code...
    
else:
    print("Starting fresh execution")
    # ...execute all operations...
```

### Using with Progress Manager

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager
import logging
from pathlib import Path

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A_profiling",
    task_type="profiling",
    logger=logger,
    total_operations=3
)

# Create context manager with progress tracking
context_manager = create_task_context_manager(
    task_id="t_1A_profiling",
    task_dir=Path("DATA/processed/t_1A_profiling"),
    progress_manager=progress_manager
)

# Check if we can resume (progress will be shown)
can_resume, checkpoint_name = context_manager.can_resume_execution()

if can_resume:
    # Restore state (progress will be shown)
    state = context_manager.restore_execution_state(checkpoint_name)
    
    # Continue execution with progress tracking
    with progress_manager.create_operation_context(
        name="continue_execution",
        total=100,
        description="Continuing execution from checkpoint"
    ) as progress:
        # ... execution code with progress updates ...
        progress.update(50, {"stage": "processing"})
        
        # Create checkpoint with progress tracking
        checkpoint_name = context_manager.create_automatic_checkpoint(
            operation_index=2,
            metrics={"processed_items": 500}
        )
        
        progress.update(50, {"stage": "completed", "checkpoint": checkpoint_name})
```

### Error Recovery

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager, CheckpointError
from pathlib import Path

# Create context manager
context_manager = create_task_context_manager(
    task_id="t_1A_profiling", 
    task_dir=Path("DATA/processed/t_1A_profiling")
)

try:
    # Check for previous error checkpoints
    checkpoints = context_manager.get_checkpoints()
    error_checkpoints = [name for name, _ in checkpoints if name.startswith("error_")]
    
    if error_checkpoints:
        print(f"Found error checkpoint: {error_checkpoints[0]}")
        
        # Restore from error checkpoint for analysis
        error_state = context_manager.restore_execution_state(error_checkpoints[0])
        
        # Extract error information
        error_info = error_state.get("error_info", {})
        print(f"Previous error: {error_info.get('message', 'Unknown error')}")
        
        # Decide how to handle the error
        if error_info.get("type") == "FileNotFoundError":
            print("Fixing file path issue and retrying...")
            # ... fix the issue ...
        else:
            print("Unknown error, starting fresh...")
            # ... start fresh execution ...
    
    # Continue with normal execution
    # ...
    
except CheckpointError as e:
    print(f"Checkpoint error: {e}")
    # Handle checkpoint-specific errors
    
except Exception as e:
    print(f"Execution error: {e}")
    # Any unhandled exception will trigger an error checkpoint in __exit__
```

### Checkpoint Management

```python
from pamola_core.utils.tasks.context_manager import create_task_context_manager
from pathlib import Path

# Create context manager
context_manager = create_task_context_manager(
    task_id="t_1A_profiling", 
    task_dir=Path("DATA/processed/t_1A_profiling")
)

# List all checkpoints
checkpoints = context_manager.get_checkpoints()
print(f"Found {len(checkpoints)} checkpoints:")
for name, timestamp in checkpoints:
    print(f"- {name} (created: {timestamp})")

# Clean up old checkpoints
if len(checkpoints) > 5:
    removed_count = context_manager.cleanup_old_checkpoints(max_checkpoints=5)
    print(f"Removed {removed_count} old checkpoints")
    
    # List remaining checkpoints
    remaining = context_manager.get_checkpoints()
    print(f"Remaining checkpoints: {[name for name, _ in remaining]}")
```

## Integration with BaseTask

The `context_manager.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.context_manager = create_task_context_manager(
    task_id=self.task_id,
    task_dir=self.task_dir,
    progress_manager=self.progress_manager
)

# Check if we can resume from a checkpoint
resumable, checkpoint_name = self.context_manager.can_resume_execution()
if resumable:
    try:
        # Restore state from checkpoint
        self._restored_state = self.context_manager.restore_execution_state(checkpoint_name)
        self.logger.info(f"Restored execution state from checkpoint: {checkpoint_name}")
        # Resumption logic handled in execute()
    except Exception as e:
        self.logger.warning(f"Could not restore from checkpoint: {e}")

# In execute() method
# Use _restored_state to determine which operations to skip
if self._restored_state is not None:
    last_completed_index = self._restored_state.get("operation_index", -1)
    # Skip operations that were already completed
    if last_completed_index >= 0:
        for i in range(last_completed_index + 1):
            # Skip operation i
            pass
        start_idx = last_completed_index + 1
    else:
        start_idx = 0
else:
    start_idx = 0

# During operation execution - create automatic checkpoints
checkpoint_name = self.context_manager.create_automatic_checkpoint(
    operation_index=i,
    metrics=self.metrics
)

# On task completion - clean up old checkpoints
self.context_manager.cleanup_old_checkpoints()
```

## Thread Safety Considerations

The context manager uses file locks to ensure thread safety when multiple processes or threads might access checkpoints simultaneously:

```python
# File lock used to prevent race conditions
with filelock.FileLock(self.lock_file, timeout=DEFAULT_LOCK_TIMEOUT):
    # Critical section for checkpoint file operations
    # ...
```

Key considerations:
1. **Timeout Handling**: The lock has a timeout (default 10 seconds) to prevent indefinite blocking
2. **Error Recovery**: Lock files are automatically released when the context exits, even after errors
3. **Shared File System**: Works across processes and different Python interpreters
4. **Exclusive Access**: Only one thread/process can access checkpoints at a time

## Best Practices

1. **Use Automatic Checkpointing**: Let the context manager handle checkpoints automatically at appropriate points in task execution.

2. **Keep State Size Manageable**: Avoid storing large data in state - only track essential execution metadata to ensure optimal performance.

3. **Handle Resumption Gracefully**: When resuming from a checkpoint, properly verify the state and skip already completed operations.

4. **Clean Up Old Checkpoints**: Periodically call `cleanup_old_checkpoints()` to manage disk space usage.

5. **Use with Progress Manager**: Integrate with a progress manager to provide visual feedback during checkpoint operations.

6. **Error Handling**: Handle checkpoint errors gracefully - restore failures shouldn't crash the entire task.

7. **Atomic Operations**: Use the context manager's atomic write functionality instead of direct file operations.

8. **State Structure**: Maintain a consistent state structure with appropriate timestamps and version information.

9. **Security Validation**: Ensure all file paths are validated before use to prevent path traversal issues.

10. **Explicit Cleanup**: Always call `cleanup()` or use the context manager pattern to ensure proper resource release.

## Checkpoint Naming Conventions

The context manager uses the following naming conventions for checkpoints:

1. **Automatic Checkpoints**: `auto_{task_id}_{operation_index}_{timestamp}`
2. **Manual Checkpoints**: User-provided name or generated name
3. **Error Checkpoints**: `error_{task_id}_{timestamp}`

All checkpoint names are sanitized to ensure they are safe to use as filenames.

## Performance Considerations

1. **State Size**: Large state dictionaries consume more disk space and take longer to serialize/deserialize. Keep state minimal.

2. **Checkpointing Frequency**: Creating checkpoints too frequently can impact performance. Use strategic points like between operations.

3. **Automatic Cleanup**: The context manager can automatically prune old checkpoints to manage disk space.

4. **Atomic Operations**: All file operations are designed to be atomic to prevent corruption during system crashes.

5. **Progress Tracking**: When using progress tracking, there is a small overhead for visual feedback, but the benefits usually outweigh the cost.