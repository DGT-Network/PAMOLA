# TaskUtils Module Documentation

## Overview

The `task_utils.py` module provides a collection of utility functions for working with tasks in the PAMOLA Core framework. These utilities simplify common operations such as directory management, path resolution, data source preparation, and execution time formatting. The module serves as a central repository for helper functions that are used across multiple task implementations, promoting code reuse and consistency.

## Key Features

- **Directory Management**: Creating and organizing standard task directories
- **Path Resolution and Security**: Secure path handling and validation
- **Data Source Preparation**: Utilities for creating data sources from file paths
- **Time Formatting**: Conversion of execution times to human-readable formats
- **Previous Task Output Detection**: Finding outputs from previous task executions
- **Secure Directory Creation**: Creating directories with proper permissions
- **Error Formatting**: Standardizing error information for reporting
- **Resource Cleanup**: Managing temporary resources

## Dependencies

- `logging`: Logging functionality
- `os`: Operating system interface
- `datetime`: Date and time handling
- `pathlib.Path`: Object-oriented filesystem paths
- `pamola_core.utils.io`: File I/O utilities, including directory creation and timestamped filenames
- `pamola_core.utils.ops.op_data_source`: Data source abstraction
- `pamola_core.utils.progress`: Progress tracking for operations
- `pamola_core.utils.tasks.task_config`: Path security validation

## Functions

### create_task_directories

```python
def create_task_directories(task_dir: Path) -> Dict[str, Path]
```

Creates standard directories for a task.

**Parameters:**
- `task_dir`: Base directory for the task

**Returns:**
- Dictionary with paths to standard directories

**Raises:**
- `ValueError`: If the task directory path is insecure

### prepare_data_source_from_paths

```python
def prepare_data_source_from_paths(file_paths: Dict[str, str],
                                   show_progress: bool = True) -> DataSource
```

Prepares a data source from file paths.

**Parameters:**
- `file_paths`: Dictionary mapping dataset names to file paths
- `show_progress`: Whether to show a progress bar during loading

**Returns:**
- DataSource with file paths added

**Raises:**
- `ValueError`: If any input path is insecure

### format_execution_time

```python
def format_execution_time(seconds: float) -> str
```

Formats execution time in seconds to a human-readable string.

**Parameters:**
- `seconds`: Execution time in seconds

**Returns:**
- Formatted execution time string (e.g., "42.5 seconds", "1 minutes, 30 seconds")

### get_artifact_path

```python
def get_artifact_path(task_dir: Path,
                      artifact_name: str,
                      artifact_type: str = "json",
                      sub_dir: str = "output",
                      include_timestamp: bool = True) -> Path
```

Gets a standardized path for a task artifact.

**Parameters:**
- `task_dir`: Base directory for the task
- `artifact_name`: Name of the artifact
- `artifact_type`: Type/extension of the artifact
- `sub_dir`: Subdirectory for the artifact
- `include_timestamp`: Whether to include a timestamp in the filename

**Returns:**
- Path to the artifact

**Raises:**
- `ValueError`: If the task directory or artifact name is insecure

### find_previous_output

```python
def find_previous_output(task_id: str,
                         data_repository: Optional[Path] = None,
                         project_root: Optional[Path] = None,
                         file_pattern: Optional[str] = None) -> List[Path]
```

Finds output files from a previous task.

**Parameters:**
- `task_id`: ID of the previous task
- `data_repository`: Path to the data repository (optional)
- `project_root`: Path to the project root (optional)
- `file_pattern`: Glob pattern to match specific files (optional)

**Returns:**
- List of paths to output files

**Raises:**
- `ValueError`: If the task ID or file pattern is insecure

### find_task_report

```python
def find_task_report(task_id: str,
                     data_repository: Optional[Path] = None,
                     project_root: Optional[Path] = None) -> Optional[Path]
```

Finds the report file from a previous task.

**Parameters:**
- `task_id`: ID of the previous task
- `data_repository`: Path to the data repository (optional)
- `project_root`: Path to the project root (optional)

**Returns:**
- Path to the report file or None if not found

**Raises:**
- `ValueError`: If the task ID is insecure

### get_temp_dir

```python
def get_temp_dir(task_dir: Path) -> Path
```

Gets a temporary directory for the task.

**Parameters:**
- `task_dir`: Base directory for the task

**Returns:**
- Path to the temporary directory

**Raises:**
- `ValueError`: If the task directory is insecure

### clean_temp_dir

```python
def clean_temp_dir(task_dir: Path) -> bool
```

Cleans the temporary directory for the task.

**Parameters:**
- `task_dir`: Base directory for the task

**Returns:**
- True if cleaning was successful, False otherwise

**Raises:**
- `ValueError`: If the task directory is insecure

### format_error_for_report

```python
def format_error_for_report(error: Exception) -> Dict[str, Any]
```

Formats an exception for inclusion in a task report.

**Parameters:**
- `error`: Exception to format

**Returns:**
- Dictionary with formatted error information

### ensure_secure_directory

```python
def ensure_secure_directory(path: Union[str, Path]) -> Path
```

Creates a directory with secure permissions.

**Parameters:**
- `path`: Path to the directory

**Returns:**
- Path to the created directory

**Raises:**
- `ValueError`: If the directory path is insecure

### is_master_key_exposed

```python
def is_master_key_exposed() -> bool
```

Checks if the master encryption key has insecure permissions.

**Returns:**
- True if the master key has insecure permissions, False otherwise

### extract_previous_output_info

```python
def extract_previous_output_info(task_id: str,
                                 data_repository: Optional[Path] = None) -> Dict[str, Any]
```

Extracts information about outputs from a previous task.

**Parameters:**
- `task_id`: ID of the previous task
- `data_repository`: Path to the data repository (optional)

**Returns:**
- Dictionary with information about previous outputs

**Raises:**
- `ValueError`: If the task ID is insecure

## Directory Structure

The module works with the standard PAMOLA directory structure:

```
PROJECT_ROOT/
├── configs/                 # Configuration files
│   ├── prj_config.json      # Project configuration
│   └── {task_id}.json       # Task configuration
├── logs/                    # Log files
└── DATA/                    # Data repository
    ├── raw/                 # Raw input data
    ├── processed/           # Processed data
    │   └── {task_id}/       # Task-specific directory
    │       ├── output/      # Task outputs
    │       ├── dictionaries/# Extracted dictionaries
    │       ├── visualizations/# Data visualizations
    │       ├── metrics/     # Metric files
    │       └── temp/        # Temporary files
    └── reports/             # Task reports
```

## Usage Examples

### Creating Task Directories

```python
from pamola_core.utils.tasks.task_utils import create_task_directories
from pathlib import Path

# Create standard task directories
task_dir = Path("DATA/processed/t_1A_profiling")
directories = create_task_directories(task_dir)

# Access specific directories
output_dir = directories["output"]
dict_dir = directories["dictionaries"]
viz_dir = directories["visualizations"]

print(f"Output directory: {output_dir}")
print(f"Dictionaries directory: {dict_dir}")
print(f"Visualizations directory: {viz_dir}")
```

### Preparing a Data Source

```python
from pamola_core.utils.tasks.task_utils import prepare_data_source_from_paths

# Define file paths
file_paths = {
    "customers": "DATA/raw/customers.csv",
    "products": "DATA/raw/products.csv",
    "transactions": "DATA/raw/transactions.csv"
}

# Create data source with progress indicator
data_source = prepare_data_source_from_paths(file_paths, show_progress=True)

# Use the data source
for dataset_name in data_source.get_dataset_names():
    df = data_source.get_dataset(dataset_name)
    print(f"Dataset {dataset_name}: {len(df)} rows, {len(df.columns)} columns")
```

### Formatting Execution Time

```python
from pamola_core.utils.tasks.task_utils import format_execution_time
import time

# Record execution time
start_time = time.time()

# Execute a task
for i in range(1000000):
    pass

# Calculate and format execution time
execution_time = time.time() - start_time
formatted_time = format_execution_time(execution_time)

print(f"Task executed in {formatted_time}")
```

### Getting an Artifact Path

```python
from pamola_core.utils.tasks.task_utils import get_artifact_path
from pathlib import Path

# Get task directory
task_dir = Path("DATA/processed/t_1A_profiling")

# Get paths for different artifact types
stats_path = get_artifact_path(
    task_dir=task_dir,
    artifact_name="field_statistics",
    artifact_type="json",
    sub_dir="output",
    include_timestamp=True
)

viz_path = get_artifact_path(
    task_dir=task_dir,
    artifact_name="distribution",
    artifact_type="png",
    sub_dir="visualizations",
    include_timestamp=True
)

dict_path = get_artifact_path(
    task_dir=task_dir,
    artifact_name="field_mappings",
    artifact_type="json",
    sub_dir="dictionaries",
    include_timestamp=False
)

print(f"Stats path: {stats_path}")
print(f"Visualization path: {viz_path}")
print(f"Dictionary path: {dict_path}")
```

### Finding Previous Task Outputs

```python
from pamola_core.utils.tasks.task_utils import find_previous_output, find_task_report
from pathlib import Path

# Find all outputs from a previous task
previous_outputs = find_previous_output(
    task_id="t_1I_import",
    data_repository=Path("DATA"),
    file_pattern="*.csv"
)

print(f"Found {len(previous_outputs)} outputs from previous task")
for output in previous_outputs:
    print(f"  - {output.name}")

# Find a specific task report
report_path = find_task_report(
    task_id="t_1I_import",
    data_repository=Path("DATA")
)

if report_path:
    print(f"Found task report: {report_path}")
else:
    print("Task report not found")
```

### Working with Temporary Directories

```python
from pamola_core.utils.tasks.task_utils import get_temp_dir, clean_temp_dir
from pathlib import Path
import os

# Get temporary directory
task_dir = Path("DATA/processed/t_1A_profiling")
temp_dir = get_temp_dir(task_dir)

# Create temporary files
temp_file1 = temp_dir / "temp1.csv"
temp_file2 = temp_dir / "temp2.json"

with open(temp_file1, "w") as f:
    f.write("temp,data\n1,2\n3,4")

with open(temp_file2, "w") as f:
    f.write('{"temp": "data"}')

print(f"Created temporary files in {temp_dir}")

# Clean up temporary files when done
clean_result = clean_temp_dir(task_dir)
print(f"Cleanup successful: {clean_result}")
```

### Creating Secure Directories

```python
from pamola_core.utils.tasks.task_utils import ensure_secure_directory
from pathlib import Path

# Create a secure directory for sensitive data
secure_dir = ensure_secure_directory(Path("DATA/sensitive"))

print(f"Created secure directory: {secure_dir}")
print(f"Directory exists: {secure_dir.exists()}")

# On Unix systems, check permissions
import os
if os.name == "posix":
    import stat
    mode = secure_dir.stat().st_mode
    is_owner_only = (mode & (stat.S_IRWXG | stat.S_IRWXO)) == 0
    print(f"Directory has owner-only permissions: {is_owner_only}")
```

### Extracting Information from Previous Task Reports

```python
from pamola_core.utils.tasks.task_utils import extract_previous_output_info
from pathlib import Path

# Extract information about previous task outputs
output_info = extract_previous_output_info(
    task_id="t_1I_import",
    data_repository=Path("DATA")
)

if output_info:
    print(f"Previous task ID: {output_info['task_id']}")
    print(f"Execution time: {output_info['execution_time']} seconds")
    print(f"Status: {output_info['status']}")
    
    # Print artifacts by type
    for artifact_type, artifacts in output_info['artifacts'].items():
        print(f"\nArtifacts of type '{artifact_type}':")
        for artifact in artifacts:
            print(f"  - {artifact['filename']}: {artifact['description']}")
else:
    print("No information found for previous task")
```

### Formatting Errors for Reports

```python
from pamola_core.utils.tasks.task_utils import format_error_for_report

try:
    # Simulate an error
    result = 1 / 0
except Exception as e:
    # Format the error for inclusion in a report
    error_info = format_error_for_report(e)
    
    print(f"Error type: {error_info['type']}")
    print(f"Error message: {error_info['message']}")
    print(f"Timestamp: {error_info['timestamp']}")
    print(f"Traceback available: {'traceback' in error_info}")
```

## Best Practices

1. **Path Security**: Always validate paths using `validate_path_security()` to prevent path traversal attacks.

2. **Standardized Directories**: Use `create_task_directories()` to ensure consistent directory structure across tasks.

3. **Timestamped Artifacts**: Use `get_artifact_path()` with `include_timestamp=True` for artifacts that may be regenerated.

4. **Resource Cleanup**: Always clean temporary resources with `clean_temp_dir()` after use.

5. **Data Source Preparation**: Use `prepare_data_source_from_paths()` for consistent data loading.

6. **Previous Task Integration**: Use `find_previous_output()` and `find_task_report()` to build task pipelines.

7. **Secure Storage**: Use `ensure_secure_directory()` for directories containing sensitive data.

8. **Error Reporting**: Use `format_error_for_report()` for consistent error reporting across tasks.

9. **Task Reuse**: Extract information from previous tasks with `extract_previous_output_info()` to avoid redundant processing.

10. **Progress Indication**: Enable progress tracking when working with multiple files or large datasets.

## Security Considerations

The module includes several security features:

- **Path Validation**: All functions validate paths using `validate_path_security()` to prevent path traversal.

- **Secure Directories**: The `ensure_secure_directory()` function creates directories with appropriate permissions.

- **Master Key Exposure Check**: The `is_master_key_exposed()` function checks if encryption keys have proper permissions.

- **Error Information Sanitization**: The `format_error_for_report()` function ensures errors don't expose sensitive details.

## Performance Considerations

For operations with large datasets or many files:

- **Progress Tracking**: Enable `show_progress=True` when preparing data sources with many files.

- **Temporary Directory Management**: Use temporary directories for intermediate results, cleaning them afterward.

- **File Filtering**: Use `file_pattern` with `find_previous_output()` to limit file scanning.

- **Path Resolution Caching**: Cache resolved paths when repeatedly accessing the same directories.