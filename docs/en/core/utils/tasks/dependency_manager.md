# TaskDependencyManager Module Documentation

## Overview

The `dependency_manager.py` module provides functionality for managing dependencies between tasks in the PAMOLA Core framework. It enables tasks to locate dependency outputs, validate dependency completion status, and access reports from dependency tasks. This module is essential for building complex task pipelines where subsequent tasks depend on the successful completion of previous tasks.

## Key Features

- **Dependency Path Resolution**: Resolves dependency paths (absolute or task-relative)
- **Dependency Validation**: Checks if dependencies have completed successfully
- **Security Validation**: Ensures dependency paths meet security requirements
- **Flexible Dependency Types**: Supports both task-based and absolute path dependencies
- **Dependency Metrics Access**: Provides access to metrics from dependency reports
- **Error Handling**: Provides clear error messages for dependency issues
- **Flexible Continuation**: Supports continue-on-error mode for non-critical dependencies

## Dependencies

- `json`, `logging`, `os`: Standard library modules
- `pathlib.Path`: Path handling
- `pamola_core.utils.io`: File I/O utilities
- `pamola_core.utils.tasks.path_security`: Path security validation

## Exception Classes

- **DependencyError**: Base exception for dependency-related errors
- **DependencyMissingError**: Exception raised when a required dependency is missing
- **DependencyFailedError**: Exception raised when a dependency task has failed

## Main Class

### TaskDependencyManager

#### Description

Manager for task dependencies. This class provides functionality for accessing and validating dependencies between tasks, including file paths, report status, and completion validation.

#### Constructor

```python
def __init__(self, task_config, logger: logging.Logger)
```

**Parameters:**
- `task_config`: Task configuration containing dependency information
- `logger`: Logger for tracking dependency operations

#### Key Attributes

- `config`: Task configuration containing dependency information
- `logger`: Logger for tracking dependency operations

#### Methods

##### get_dependency_output

```python
def get_dependency_output(
    self, 
    dependency_id: str, 
    file_pattern: Optional[str] = None
) -> Union[Path, List[Path]]
```

Get the output directory or files from a dependency.

**Parameters:**
- `dependency_id`: Dependency ID (task ID) or absolute path
- `file_pattern`: Optional file pattern to match within the dependency output dir

**Returns:**
- Path to the dependency output directory or list of matching files

**Raises:**
- `PathSecurityError`: If the path fails security validation
- `DependencyMissingError`: If the dependency output directory doesn't exist

##### get_dependency_report

```python
def get_dependency_report(self, dependency_id: str) -> Path
```

Get the report file from a dependency.

**Parameters:**
- `dependency_id`: Dependency ID (task ID)

**Returns:**
- Path to the dependency report file

**Raises:**
- `DependencyMissingError`: If the dependency report doesn't exist
- `ValueError`: If trying to get a report for an absolute dependency path

##### assert_dependencies_completed

```python
def assert_dependencies_completed(self) -> bool
```

Check if all dependencies have completed successfully.

**Returns:**
- True if all dependencies are complete, False otherwise

**Raises:**
- `DependencyMissingError`: If a dependency report is missing
- `DependencyFailedError`: If a dependency task has failed

##### is_absolute_dependency

```python
def is_absolute_dependency(self, dependency_id: str) -> bool
```

Check if a dependency ID represents an absolute path.

**Parameters:**
- `dependency_id`: Dependency ID to check

**Returns:**
- True if dependency_id represents an absolute path

##### get_dependency_metrics

```python
def get_dependency_metrics(
    self, 
    dependency_id: str, 
    metric_path: Optional[str] = None
) -> Dict[str, Any]
```

Get metrics from a dependency report.

**Parameters:**
- `dependency_id`: Dependency ID (task ID)
- `metric_path`: Optional path within metrics to extract (dot notation)

**Returns:**
- Dictionary of metrics from the dependency report

**Raises:**
- `DependencyMissingError`: If the dependency report doesn't exist
- `KeyError`: If the specified metric path doesn't exist

##### get_dependency_status

```python
def get_dependency_status(self, dependency_id: str) -> Dict[str, Any]
```

Get status information about a dependency.

**Parameters:**
- `dependency_id`: Dependency ID (task ID)

**Returns:**
- Dictionary with status information

**Raises:**
- `DependencyMissingError`: If the dependency report doesn't exist

## Dependency Resolution Logic

The module provides two ways to identify dependencies:

1. **Task-based Dependencies**: Using a task ID (e.g., "t_1A_profiling"), which resolves to that task's output directory
2. **Absolute Path Dependencies**: Using an absolute file path, which points directly to input data

Task-based dependencies are checked for completion by examining their report files, while absolute path dependencies are only checked for existence.

### Path Resolution Algorithm

When resolving `dependency_id`:

1. Check if it contains path separators or looks like an absolute path
2. If it's an absolute path:
   - Validate path security
   - Return the path, optionally with matching files if a pattern is provided
3. If it's a task ID:
   - Use the project's directory structure logic to find the task's output directory
   - Return the output directory, optionally with matching files if a pattern is provided

### Completion Validation Algorithm

When validating task dependencies:

1. Skip absolute path dependencies (they only need to exist, not "complete")
2. For each task dependency:
   - Locate its report file
   - Load and parse the report
   - Check the `success` flag in the report
   - If not successful, raise an error or warn (based on `continue_on_error` setting)

## Usage Examples

### Accessing Dependency Outputs

```python
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Mock configuration with dependencies
class MockConfig:
    dependencies = ["t_1A_ingestion", "t_1B_profiling"]
    continue_on_error = False
    allowed_external_paths = ["/data/external"]
    allow_external = True
    
    def get_task_output_dir(self, task_id):
        return Path(f"DATA/processed/{task_id}/output")
    
    def get_reports_dir(self):
        return Path("DATA/reports")

# Create dependency manager
dependency_manager = TaskDependencyManager(MockConfig(), logger)

# Get output directory of a dependency task
output_dir = dependency_manager.get_dependency_output("t_1A_ingestion")
print(f"Dependency output directory: {output_dir}")

# Get specific files from a dependency task
csv_files = dependency_manager.get_dependency_output("t_1A_ingestion", "*.csv")
print(f"Found {len(csv_files)} CSV files")
for file_path in csv_files:
    print(f"- {file_path}")

# Use an absolute path as a dependency
external_data = dependency_manager.get_dependency_output("/data/external/reference.csv")
print(f"External data path: {external_data}")
```

### Validating Dependencies

```python
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager, DependencyError
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create dependency manager
dependency_manager = TaskDependencyManager(task_config, logger)

# Check if all dependencies have completed successfully
try:
    dependencies_ok = dependency_manager.assert_dependencies_completed()
    if dependencies_ok:
        print("All dependencies have completed successfully, proceeding with task")
        # Continue with task execution
    else:
        print("Dependencies check failed")
except DependencyError as e:
    print(f"Dependency error: {e}")
    # Handle dependency error
    exit(1)
```

### Accessing Dependency Metrics

```python
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager, DependencyMissingError
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create dependency manager
dependency_manager = TaskDependencyManager(task_config, logger)

try:
    # Get all metrics from a dependency
    metrics = dependency_manager.get_dependency_metrics("t_1A_profiling")
    print(f"Dependency metrics: {metrics}")
    
    # Get a specific metric using dot notation
    field_count = dependency_manager.get_dependency_metrics(
        "t_1A_profiling", 
        "field_statistics.count"
    )
    print(f"Field count: {field_count}")
    
    # Get dependency execution status
    status = dependency_manager.get_dependency_status("t_1A_profiling")
    print(f"Dependency status: {status['success']}")
    print(f"Execution time: {status['execution_time']} seconds")
    
except DependencyMissingError as e:
    print(f"Dependency error: {e}")
    # Handle missing dependency
except KeyError as e:
    print(f"Metric not found: {e}")
    # Handle missing metric
```

### Working with Absolute Path Dependencies

```python
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager, PathSecurityError
import logging
from pathlib import Path

# Set up logger
logger = logging.getLogger("task.example")

# Create dependency manager
dependency_manager = TaskDependencyManager(task_config, logger)

# Check if a dependency is an absolute path
dependency_id = "/data/external/reference_data/customers.csv"
is_absolute = dependency_manager.is_absolute_dependency(dependency_id)
print(f"Is absolute path: {is_absolute}")

# Get matching files from an absolute path dependency
try:
    data_files = dependency_manager.get_dependency_output(
        "/data/external/reference_data", 
        "*.csv"
    )
    print(f"Found {len(data_files)} CSV files")
    
except PathSecurityError as e:
    print(f"Security error: {e}")
    # Handle security error - path might be outside allowed areas
    
except DependencyMissingError as e:
    print(f"Missing dependency: {e}")
    # Handle missing dependency directory
```

### Continue-on-Error Mode

```python
from pamola_core.utils.tasks.dependency_manager import TaskDependencyManager
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Configuration with continue_on_error enabled
class ContinueOnErrorConfig:
    dependencies = ["t_1A_ingestion", "t_1B_profiling"]
    continue_on_error = True  # This is the key setting
    
    def get_task_output_dir(self, task_id):
        return Path(f"DATA/processed/{task_id}/output")
    
    def get_reports_dir(self):
        return Path("DATA/reports")

# Create dependency manager
dependency_manager = TaskDependencyManager(ContinueOnErrorConfig(), logger)

# This will log warnings instead of raising exceptions for missing dependencies
result = dependency_manager.assert_dependencies_completed()
print(f"Dependencies check result: {result}")

# Similarly, getting outputs from missing dependencies won't raise exceptions
try:
    missing_output = dependency_manager.get_dependency_output("non_existent_task")
    print(f"Output path (even if missing): {missing_output}")
except Exception as e:
    print(f"This won't be raised in continue-on-error mode: {e}")
```

## Integration with BaseTask

The `dependency_manager.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.dependency_manager = TaskDependencyManager(
    task_config=self.config,
    logger=self.logger
)

# Check dependencies using the dependency manager
if not self.dependency_manager.assert_dependencies_completed():
    raise TaskDependencyError(f"Task dependencies not satisfied for {self.task_id}")

# In BaseTask methods, access dependency outputs
dependency_output = self.dependency_manager.get_dependency_output(
    dependency_id, 
    file_pattern="*.csv"
)

# Get metrics from dependencies for use in current task
metrics = self.dependency_manager.get_dependency_metrics(dependency_id)
```

## Error Handling

The module provides a hierarchy of exceptions to help with error handling:

1. **DependencyError**: Base class for all dependency-related errors
   - **DependencyMissingError**: Raised when a dependency is missing
   - **DependencyFailedError**: Raised when a dependency task has failed

Additionally, **PathSecurityError** can be raised if a dependency path fails security validation.

## Configuration Requirements

The `task_config` object provided to the `TaskDependencyManager` constructor should implement the following attributes and methods:

```python
class TaskConfig:
    # List of dependency task IDs
    dependencies: List[str]
    
    # Whether to continue on dependency errors
    continue_on_error: bool
    
    # Security settings for external paths
    allowed_external_paths: List[Path]
    allow_external: bool
    
    def get_task_output_dir(self, task_id: str) -> Path:
        """Get the output directory for a task."""
        ...
    
    def get_reports_dir(self) -> Path:
        """Get the directory containing task reports."""
        ...
```

## Security Considerations

The module implements several security features:

1. **Path Validation**: All dependency paths are validated through the `validate_path_security` function to prevent path traversal attacks.

2. **Absolute Path Control**: The `allowed_external_paths` and `allow_external` settings control whether absolute paths outside the project directory are permitted.

3. **Report Access Control**: Reports can only be accessed for task dependencies, not for absolute path dependencies, to prevent information leakage.

## Dependency Types

### Task-Based Dependencies (Internal)

These are task IDs that resolve to specific task output directories within the project structure. Typical usage:

```python
output_dir = dependency_manager.get_dependency_output("t_1A_ingestion")
```

The output directory path is constructed using `task_config.get_task_output_dir(dependency_id)`.

### Absolute Path Dependencies (External)

These are full paths to external data sources. Typical usage:

```python
external_data = dependency_manager.get_dependency_output("/data/external/reference.csv")
```

These paths must be:
1. Validated for security
2. Included in `allowed_external_paths` if `allow_external` is not enabled

## Best Practices

1. **Use Task IDs for Internal Dependencies**: For data flows within your project, use task IDs as dependencies to maintain logical connections.

2. **Use Absolute Paths Judiciously**: Only use absolute paths for truly external data that isn't produced by your task pipeline.

3. **Validate All Dependencies Early**: Call `assert_dependencies_completed()` at task initialization to fail fast if dependencies aren't met.

4. **Use Specific File Patterns**: When getting dependency outputs, provide specific file patterns to get only the files you need.

5. **Handle Missing Dependencies Gracefully**: Use try-except blocks to handle missing dependencies with clear error messages.

6. **Use Metrics Access**: Leverage `get_dependency_metrics()` to access metadata from previous tasks, avoiding duplicate calculations.

7. **Consider continue_on_error**: For non-critical dependencies, consider enabling continue_on_error in your task configuration.

8. **Specify Metric Paths**: When accessing dependency metrics, use dot notation to get specific values rather than retrieving all metrics.

9. **Check Dependency Status**: Use `get_dependency_status()` to check execution details beyond simple success/failure.

10. **Secure External Paths**: Always set appropriate `allowed_external_paths` in your configuration to control external data access.