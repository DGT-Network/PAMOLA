# ExecutionLog Module Documentation

## Overview

The `execution_log.py` module provides functionality for managing persistent task execution history in the PAMOLA Core framework. It enables tracking of task dependencies, data flow, and execution status at the project level. This module is essential for maintaining a comprehensive execution history across multiple task runs, facilitating workflow tracking, and enabling dependency validation between tasks.

## Key Features

- **Execution Log Management**: Initializes and maintains a centralized execution log
- **Task Execution Recording**: Captures comprehensive metadata about task executions
- **Data Flow Tracking**: Tracks input and output dependencies between tasks
- **Execution History Querying**: Enables retrieval of past execution information
- **Task Dependency Validation**: Validates task dependencies based on execution history
- **Thread-Safe Operations**: Ensures concurrent access doesn't corrupt the log
- **Path Security Validation**: Ensures all file paths meet security requirements
- **Integration with Progress Tracking**: Coordinates with progress manager for operation visibility
- **Maintenance Utilities**: Tools for cleaning up and validating the execution log

## Dependencies

- `logging`, `os`, `time`, `uuid`: Standard library modules
- `datetime`: For timestamp management
- `pathlib.Path`: Path manipulation
- `typing`: Type annotations
- `filelock`: For thread-safe access to the execution log
- `pamola_core.utils.io`: File I/O utilities
- `pamola_core.utils.tasks.task_config`: Project root finding and path validation

## Exception Classes

### ExecutionLogError

Base exception raised for execution log errors. This exception is used for all errors related to execution log operations.

## Protocols

### ProgressManagerProtocol

Protocol defining the interface for progress managers used by the execution log module.

## Constants

- `DEFAULT_EXECUTION_LOG_PATH`: Default path for the execution log file (`configs/execution_log.json`)
- `DEFAULT_LOCK_TIMEOUT`: Default lock timeout in seconds (10s)

## Functions

### _get_execution_log_path

```python
def _get_execution_log_path() -> Path
```

Get the path to the execution log file.

**Returns:**
- Path to the execution log file

### initialize_execution_log

```python
def initialize_execution_log(
    project_path: Optional[Path] = None, 
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Path
```

Initialize the execution log file. Creates or initializes the execution log file with empty data.

**Parameters:**
- `project_path`: Path to the project root (optional, auto-detected if None)
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Path to the execution log file

**Raises:**
- `ExecutionLogError`: If initialization fails

### record_task_execution

```python
def record_task_execution(
    task_id: str,
    task_type: str,
    success: bool,
    execution_time: float,
    report_path: Path,
    input_datasets: Optional[Dict[str, str]] = None,
    output_artifacts: Optional[List[Any]] = None,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Optional[str]
```

Record a task execution in the execution log.

**Parameters:**
- `task_id`: ID of the task
- `task_type`: Type of the task
- `success`: Whether the task executed successfully
- `execution_time`: Task execution time in seconds
- `report_path`: Path to the task report
- `input_datasets`: Dictionary of input datasets (optional)
- `output_artifacts`: List of output artifacts (optional)
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Task run UUID or None if recording fails

**Raises:**
- `ExecutionLogError`: If recording fails

### get_task_execution_history

```python
def get_task_execution_history(
    task_id: Optional[str] = None,
    limit: int = 10,
    success_only: bool = False,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> List[Dict[str, Any]]
```

Get execution history for a specific task or all tasks.

**Parameters:**
- `task_id`: ID of the task (None for all tasks)
- `limit`: Maximum number of executions to return
- `success_only`: Whether to include only successful executions
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- List of execution records

**Raises:**
- `ExecutionLogError`: If history cannot be retrieved

### find_latest_execution

```python
def find_latest_execution(
    task_id: str, 
    success_only: bool = True,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Optional[Dict[str, Any]]
```

Find the most recent execution of a task.

**Parameters:**
- `task_id`: ID of the task
- `success_only`: Whether to include only successful executions
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Execution record or None if not found

**Raises:**
- `ExecutionLogError`: If execution cannot be found

### find_task_by_output

```python
def find_task_by_output(
    file_path: Union[str, Path],
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Optional[Dict[str, Any]]
```

Find the task that produced a specific output file.

**Parameters:**
- `file_path`: Path to the output file
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Task execution record or None if not found

**Raises:**
- `ExecutionLogError`: If task cannot be found

### track_input_files

```python
def track_input_files(
    task_id: str, 
    file_paths: List[Union[str, Path]],
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> bool
```

Register input files for a task. This function updates the task's input files in the execution log, which can be used to trace data flow.

**Parameters:**
- `task_id`: ID of the task
- `file_paths`: List of input file paths
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- True if successful, False otherwise

**Raises:**
- `ExecutionLogError`: If tracking fails

### track_output_files

```python
def track_output_files(
    task_id: str, 
    file_paths: List[Union[str, Path]],
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> bool
```

Register output files from a task. This function updates the task's output files in the execution log, which can be used to trace data flow.

**Parameters:**
- `task_id`: ID of the task
- `file_paths`: List of output file paths
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- True if successful, False otherwise

**Raises:**
- `ExecutionLogError`: If tracking fails

### update_execution_record

```python
def update_execution_record(
    task_run_id: str, 
    updates: Dict[str, Any],
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> bool
```

Update an existing execution record.

**Parameters:**
- `task_run_id`: Unique ID for the task execution
- `updates`: Dictionary of updates to apply
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- True if successful, False otherwise

**Raises:**
- `ExecutionLogError`: If update fails

### remove_execution_record

```python
def remove_execution_record(
    task_run_id: str,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> bool
```

Remove an execution record from the log.

**Parameters:**
- `task_run_id`: Unique ID for the task execution
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- True if successful, False otherwise

**Raises:**
- `ExecutionLogError`: If removal fails

### cleanup_old_executions

```python
def cleanup_old_executions(
    max_age_days: int = 30,
    max_per_task: int = 10,
    dry_run: bool = False,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Tuple[int, List[str]]
```

Clean up old execution records.

**Parameters:**
- `max_age_days`: Maximum age of execution records to keep (in days)
- `max_per_task`: Maximum number of executions to keep per task
- `dry_run`: Whether to perform a dry run (don't actually delete)
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Tuple containing:
  - Number of records removed
  - List of removed task run IDs

**Raises:**
- `ExecutionLogError`: If cleanup fails

### validate_execution_log

```python
def validate_execution_log(
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Tuple[bool, List[str]]
```

Validate the execution log. Checks for inconsistencies, missing fields, and other issues.

**Parameters:**
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Tuple containing:
  - Whether the log is valid
  - List of validation errors

**Raises:**
- `ExecutionLogError`: If validation fails

### export_execution_log

```python
def export_execution_log(
    output_path: Optional[Path] = None,
    format: str = "json",
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Path
```

Export the execution log to a file.

**Parameters:**
- `output_path`: Path to save the export (optional)
- `format`: Export format ("json" or "csv")
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Path to the exported file

**Raises:**
- `ExecutionLogError`: If export fails

### import_execution_log

```python
def import_execution_log(
    input_path: Path, 
    merge: bool = False,
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> bool
```

Import an execution log from a file.

**Parameters:**
- `input_path`: Path to the file to import
- `merge`: Whether to merge with existing log (True) or replace (False)
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- True if successful, False otherwise

**Raises:**
- `ExecutionLogError`: If import fails

## Internal Functions

### _load_execution_log

```python
def _load_execution_log(
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> Dict[str, Any]
```

Load the execution log.

**Parameters:**
- `progress_manager`: Progress manager for tracking (optional)

**Returns:**
- Execution log data

**Raises:**
- `ExecutionLogError`: If log cannot be loaded

### _save_execution_log

```python
def _save_execution_log(
    data: Dict[str, Any],
    progress_manager: Optional[ProgressManagerProtocol] = None
) -> None
```

Save the execution log.

**Parameters:**
- `data`: Execution log data
- `progress_manager`: Progress manager for tracking (optional)

**Raises:**
- `ExecutionLogError`: If log cannot be saved

## Execution Log Structure

The execution log is stored as a JSON file with the following structure:

```json
{
  "tasks": {
    "t_1I": {
      "task_type": "ingestion",
      "last_execution": 1714502400,
      "last_status": "success",
      "last_report_path": "DATA/reports/t_1I_report.json",
      "last_task_run_id": "f8a7d3e1-2b56-4c9a-8f7e-3d1a2b3c4d5e"
    },
    "t_1P": {
      "task_type": "profiling",
      "last_execution": 1714505000,
      "last_status": "success",
      "last_report_path": "DATA/reports/t_1P_report.json",
      "last_task_run_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d"
    }
  },
  "executions": [
    {
      "task_id": "t_1I",
      "task_type": "ingestion",
      "task_run_id": "f8a7d3e1-2b56-4c9a-8f7e-3d1a2b3c4d5e",
      "timestamp": "2025-05-01T12:00:00",
      "success": true,
      "execution_time": 45.2,
      "report_path": "DATA/reports/t_1I_report.json",
      "input_files": [
        {
          "name": "customers",
          "path": "DATA/raw/customers.csv"
        }
      ],
      "output_files": [
        {
          "path": "DATA/processed/t_1I/output/processed_customers.csv",
          "type": "csv",
          "description": "Processed customer data"
        }
      ],
      "hostname": "analysis-server"
    },
    {
      "task_id": "t_1P",
      "task_type": "profiling",
      "task_run_id": "a1b2c3d4-e5f6-7a8b-9c0d-1e2f3a4b5c6d",
      "timestamp": "2025-05-01T13:00:00",
      "success": true,
      "execution_time": 120.5,
      "report_path": "DATA/reports/t_1P_report.json",
      "input_files": [
        {
          "name": "customers",
          "path": "DATA/processed/t_1I/output/processed_customers.csv"
        }
      ],
      "output_files": [
        {
          "path": "DATA/processed/t_1P/output/customer_profile.json",
          "type": "json",
          "description": "Customer data profile"
        },
        {
          "path": "DATA/processed/t_1P/output/profile_chart.png",
          "type": "png",
          "description": "Profile visualization"
        }
      ],
      "hostname": "analysis-server"
    }
  ],
  "timestamp": "2025-05-01T13:10:00",
  "version": "1.0.0",
  "last_modified": "2025-05-01T13:10:00"
}
```

## Usage Examples

### Recording Task Execution

```python
from pamola_core.utils.tasks.execution_log import record_task_execution
from pathlib import Path

# Record a task execution
task_run_id = record_task_execution(
    task_id="t_1A_profiling",
    task_type="profiling",
    success=True,
    execution_time=125.3,
    report_path=Path("DATA/reports/t_1A_profiling_report.json"),
    input_datasets={
        "customers": "DATA/raw/customers.csv",
        "products": "DATA/raw/products.csv"
    },
    output_artifacts=[
        # Path objects
        Path("DATA/processed/t_1A_profiling/output/field_statistics.json"),
        Path("DATA/processed/t_1A_profiling/output/profile_chart.png"),
        
        # Or dictionaries with metadata
        {
            "path": "DATA/processed/t_1A_profiling/output/summary.txt",
            "type": "text",
            "description": "Summary of profiling results"
        }
    ]
)

print(f"Task execution recorded with ID: {task_run_id}")
```

### Finding Task Execution History

```python
from pamola_core.utils.tasks.execution_log import get_task_execution_history, find_latest_execution

# Get execution history for a specific task
history = get_task_execution_history(
    task_id="t_1A_profiling",
    limit=5,
    success_only=True
)

print(f"Found {len(history)} execution records")
for execution in history:
    print(f"Run: {execution['task_run_id']}")
    print(f"Time: {execution['timestamp']}")
    print(f"Duration: {execution['execution_time']} seconds")
    print("---")

# Find the most recent execution
latest = find_latest_execution(
    task_id="t_1A_profiling", 
    success_only=True
)

if latest:
    print(f"Latest execution: {latest['timestamp']}")
    print(f"Status: {'Success' if latest['success'] else 'Failed'}")
else:
    print("No executions found")
```

### Tracking Data Flow

```python
from pamola_core.utils.tasks.execution_log import track_input_files, track_output_files, find_task_by_output
from pathlib import Path

# Track input files for a task
track_input_files(
    task_id="t_1A_profiling",
    file_paths=[
        "DATA/raw/customers.csv",
        "DATA/raw/products.csv"
    ]
)

# Track output files
track_output_files(
    task_id="t_1A_profiling",
    file_paths=[
        Path("DATA/processed/t_1A_profiling/output/field_statistics.json"),
        Path("DATA/processed/t_1A_profiling/output/profile_chart.png")
    ]
)

# Find task that produced a specific file
output_file = "DATA/processed/t_1A_profiling/output/field_statistics.json"
producing_task = find_task_by_output(output_file)

if producing_task:
    print(f"File was produced by task: {producing_task['task_id']}")
    print(f"Run at: {producing_task['timestamp']}")
else:
    print(f"No task found that produced {output_file}")
```

### Integration with Progress Manager

```python
from pamola_core.utils.tasks.execution_log import record_task_execution
from pamola_core.utils.tasks.progress_manager import create_task_progress_manager
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create progress manager
progress_manager = create_task_progress_manager(
    task_id="t_1A",
    task_type="profiling",
    logger=logger,
    total_operations=3
)

# Record task execution with progress tracking
with progress_manager.create_operation_context(
    name="record_execution",
    total=1,
    description="Recording task execution"
) as progress:
    
    task_run_id = record_task_execution(
        task_id="t_1A",
        task_type="profiling",
        success=True,
        execution_time=125.3,
        report_path=Path("DATA/reports/t_1A_report.json"),
        input_datasets={"customers": "DATA/raw/customers.csv"},
        output_artifacts=[Path("DATA/processed/t_1A/output/results.json")],
        progress_manager=progress_manager  # Pass the progress manager
    )
    
    progress.update(1, {"status": "completed", "task_run_id": task_run_id})

# Get execution history with progress tracking
with progress_manager.create_operation_context(
    name="get_history",
    total=1,
    description="Getting execution history"
) as progress:
    
    history = get_task_execution_history(
        task_id="t_1A",
        limit=5,
        progress_manager=progress_manager
    )
    
    progress.update(1, {"status": "completed", "count": len(history)})

# Clean up progress manager when done
progress_manager.close()
```

### Maintaining the Execution Log

```python
from pamola_core.utils.tasks.execution_log import (
    cleanup_old_executions, validate_execution_log, 
    export_execution_log, import_execution_log
)
from pathlib import Path

# Validate the execution log
valid, errors = validate_execution_log()
if valid:
    print("Execution log is valid")
else:
    print(f"Execution log has {len(errors)} validation errors:")
    for error in errors:
        print(f"- {error}")

# Clean up old executions
removed_count, removed_ids = cleanup_old_executions(
    max_age_days=60,  # Keep records from last 60 days
    max_per_task=5,   # Keep at most 5 executions per task
    dry_run=True      # Just simulate, don't actually delete
)
print(f"Would remove {removed_count} old execution records")

# Actually perform the cleanup
removed_count, removed_ids = cleanup_old_executions(
    max_age_days=60,
    max_per_task=5,
    dry_run=False
)
print(f"Removed {removed_count} old execution records")

# Export the execution log
export_path = export_execution_log(
    output_path=Path("backup/execution_log_backup.json"),
    format="json"
)
print(f"Execution log exported to {export_path}")

# Export as CSV for analysis
csv_export_path = export_execution_log(
    output_path=Path("analysis/execution_log.csv"),
    format="csv"
)
print(f"Execution log exported as CSV to {csv_export_path}")

# Import from backup
success = import_execution_log(
    input_path=Path("backup/execution_log_backup.json"),
    merge=False  # Replace current log
)
if success:
    print("Execution log imported successfully")
else:
    print("Failed to import execution log")
```

## Integration with BaseTask

The `execution_log.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.finalize()
# Record task execution in the execution log
try:
    record_task_execution(
        task_id=self.task_id,
        task_type=self.task_type,
        success=success,
        execution_time=self.execution_time,
        report_path=report_path,
        input_datasets=self.input_datasets,
        output_artifacts=self.artifacts,
        progress_manager=self.progress_manager
    )
except Exception as e:
    self.logger.error(f"Error recording task execution: {str(e)}")
    # Continue with cleanup despite the error
```

## Task Dependency Validation

One of the key features of the execution log is validating task dependencies. This is typically done by the `TaskDependencyManager` component, which uses the execution log to verify that dependency tasks have been executed successfully before running a task that depends on them:

```python
# In TaskDependencyManager
def assert_dependencies_completed(self) -> bool:
    """Verify that all dependencies have been completed successfully."""
    dependencies = getattr(self.config, "dependencies", [])
    if not dependencies:
        return True
    
    for dependency_id in dependencies:
        # Find the latest execution of the dependency task
        try:
            latest_execution = find_latest_execution(
                task_id=dependency_id,
                success_only=True
            )
            
            if not latest_execution:
                self.logger.error(f"Dependency task {dependency_id} has not been executed successfully")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking dependency {dependency_id}: {str(e)}")
            return False
    
    return True
```

## Thread Safety Considerations

The execution log module uses file locks to ensure thread safety when multiple processes or threads might access the log simultaneously:

```python
# File lock used to prevent race conditions
lock_path = f"{log_path}.lock"
with filelock.FileLock(lock_path, timeout=DEFAULT_LOCK_TIMEOUT):
    # Critical section for reading/writing the log
    # ...
```

Key considerations:
1. **Timeout Handling**: The lock has a timeout (default 10 seconds) to prevent indefinite blocking
2. **Error Recovery**: Lock files are automatically released when the context exits, even after errors
3. **Shared File System**: Works across processes and different Python interpreters
4. **Exclusive Access**: Only one thread/process can modify the log at a time

## Best Practices

1. **Record All Task Executions**: Always record task executions to maintain a complete history.

2. **Track Input and Output Files**: Explicitly track input and output files to enable data flow analysis.

3. **Validate Task Dependencies**: Use the execution log to validate that dependencies have been satisfied.

4. **Maintain the Log**: Periodically run cleanup to prevent the log from growing too large.

5. **Export Backups**: Regularly export the execution log for backup and analysis.

6. **Use Progress Manager**: Pass a progress manager when working with large logs for better user experience.

7. **Handle Failures Gracefully**: Always wrap execution log operations in try-except blocks to handle errors.

8. **Validate Path Security**: All file paths added to the log should pass security validation.

9. **Keep History Reasonable**: Configure `cleanup_old_executions` to maintain a manageable history.

10. **Monitor Log Size**: The execution log can grow large over time, so monitor its size and clean up as needed.

## Common Patterns

### Task Dependency Chain

```
Task A → Task B → Task C
```

In this pattern, each task depends on the output of the previous task. The execution log enables:

- Validating that Task A was executed successfully before running Task B
- Finding the outputs of Task A to use as inputs for Task B
- Tracing the complete lineage of data transformations from Task A to Task C

### Task Pipeline Visualization

The execution log provides data that can be used to visualize task pipelines:

```python
import networkx as nx
import matplotlib.pyplot as plt
from pamola_core.utils.tasks.execution_log import get_task_execution_history

# Get all task executions
history = get_task_execution_history(limit=1000)

# Create a directed graph
G = nx.DiGraph()

# Add nodes (tasks)
for execution in history:
    G.add_node(execution['task_id'], type=execution['task_type'])

# Add edges (dependencies)
for execution in history:
    task_id = execution['task_id']
    
    # Look for input files that match output files of other tasks
    for input_file in execution.get('input_files', []):
        input_path = input_file.get('path', '')
        
        # Find task that produced this file
        for other_exec in history:
            if other_exec['task_id'] == task_id:
                continue
                
            for output_file in other_exec.get('output_files', []):
                if output_file.get('path', '') == input_path:
                    # Add edge from producer to consumer
                    G.add_edge(other_exec['task_id'], task_id)

# Draw the graph
plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_color='lightblue', node_size=1500, arrows=True)
plt.title('Task Dependency Graph')
plt.savefig('task_dependency_graph.png')
```

### Execution Time Trends

The execution log can be used to analyze task performance over time:

```python
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from pamola_core.utils.tasks.execution_log import get_task_execution_history

# Get execution history for a specific task
history = get_task_execution_history(task_id='t_1A', limit=100)

# Convert to DataFrame
data = []
for execution in history:
    data.append({
        'timestamp': datetime.datetime.fromisoformat(execution['timestamp']),
        'execution_time': execution['execution_time'],
        'success': execution['success']
    })
    
df = pd.DataFrame(data)

# Plot execution time trends
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['execution_time'], marker='o')
plt.title('Task Execution Time Trends')
plt.xlabel('Date')
plt.ylabel('Execution Time (seconds)')
plt.grid(True)
plt.savefig('execution_time_trends.png')
```