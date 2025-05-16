# TaskDirectoryManager Module Documentation

## Overview

The `directory_manager.py` module provides functionality for managing task directory structures in the PAMOLA Core framework. It ensures proper directory creation, path validation, and standardized artifact path generation. This module is essential for maintaining a consistent directory organization across all tasks, enhancing security through path validation, and simplifying the management of task artifacts.

## Key Features

- **Directory Structure Management**: Creates and validates standard task directory structure
- **Artifact Path Generation**: Creates standardized paths for various types of task artifacts
- **Path Security Validation**: Ensures all paths meet security requirements
- **Temporary Directory Management**: Handles cleanup of temporary files and directories
- **Timestamped Filenames**: Supports automatic timestamping of artifact filenames
- **External File Import**: Safely imports external files into the task structure
- **Progress Tracking Integration**: Provides visual feedback during directory operations
- **Path Normalization**: Ensures consistent path formats throughout the application

## Dependencies

- `logging`, `shutil`: Standard library modules
- `datetime`: For timestamped filenames
- `pathlib.Path`: Path handling
- `pamola_core.utils.io.ensure_directory`: Directory creation utility
- `pamola_core.utils.tasks.path_security`: Path security validation

## Constants

- `DEFAULT_DIRECTORY_SUFFIXES`: Default subdirectories to create if not specified in configuration
  - `input`: For input files
  - `output`: For output files
  - `temp`: For temporary files
  - `logs`: For log files
  - `dictionaries`: For data dictionaries
  - `visualizations`: For charts and visual artifacts
  - `metrics`: For metric outputs

## Protocol Classes

### TaskConfigProtocol

Protocol defining the required interface for task configuration.

```python
class TaskConfigProtocol(Protocol):
    task_id: str
    project_root: Path

    def get_task_dir(self) -> Path: ...
    def get_reports_dir(self) -> Path: ...
```

### TaskProgressManager

Protocol defining the required interface for progress management.

```python
class TaskProgressManager(Protocol):
    def create_operation_context(self, name: str, total: int, description: Optional[str] = None,
                                unit: str = "items", leave: bool = False) -> Any: ...
    def log_info(self, message: str) -> None: ...
    def log_warning(self, message: str) -> None: ...
    def log_error(self, message: str) -> None: ...
    def log_debug(self, message: str) -> None: ...
```

## Exception Classes

- **DirectoryManagerError**: Base exception for directory manager errors
- **PathValidationError**: Exception raised when a path fails validation
- **DirectoryCreationError**: Exception raised when directory creation fails

## Main Class

### TaskDirectoryManager

#### Description

Manager for task directory structures and path resolution. This class handles the creation and management of standard task directory structures, provides path resolution for artifacts, and ensures proper path security validation.

#### Constructor

```python
def __init__(
    self,
    task_config: Any,
    logger: Optional[logging.Logger] = None,
    progress_manager: Optional[TaskProgressManager] = None
)
```

**Parameters:**
- `task_config`: Task configuration object containing directory information
- `logger`: Logger for directory operations (optional)
- `progress_manager`: Progress manager for tracking directory operations (optional)

**Raises:**
- `DirectoryManagerError`: If initialization fails

#### Key Attributes

- `config`: Task configuration containing directory information
- `logger`: Logger for directory operations
- `progress_manager`: Progress manager for tracking
- `task_id`: ID of the task
- `project_root`: Path to the project root
- `task_dir`: Path to the task directory
- `directory_suffixes`: List of directory suffixes to create
- `directories`: Dictionary mapping directory types to their paths
- `_created_directories`: Set of directories created by this manager
- `_initialized`: Flag indicating if directories have been initialized

#### Key Methods

##### ensure_directories

```python
def ensure_directories(self) -> Dict[str, Path]
```

Create and validate all required task directories.

**Returns:**
- Dictionary mapping directory types to their paths

**Raises:**
- `DirectoryCreationError`: If directory creation fails
- `PathValidationError`: If a path fails security validation

##### get_directory

```python
def get_directory(self, dir_type: str) -> Path
```

Get path to a specific directory type.

**Parameters:**
- `dir_type`: Directory type (e.g., "input", "output", "temp")

**Returns:**
- Path to the requested directory

**Raises:**
- `DirectoryManagerError`: If directory type is unknown or not created

##### get_artifact_path

```python
def get_artifact_path(
    self,
    artifact_name: str,
    artifact_type: str = "json",
    subdir: str = "output",
    include_timestamp: bool = True
) -> Path
```

Generate standardized path for an artifact.

**Parameters:**
- `artifact_name`: Name of the artifact
- `artifact_type`: Type/extension of the artifact (without dot)
- `subdir`: Subdirectory for the artifact (e.g., "output", "visualizations")
- `include_timestamp`: Whether to include a timestamp in the filename

**Returns:**
- Path to the artifact

**Raises:**
- `PathValidationError`: If artifact path fails validation
- `DirectoryManagerError`: If subdirectory does not exist

##### clean_temp_directory

```python
def clean_temp_directory(self) -> bool
```

Clean temporary files and directories.

**Returns:**
- True if cleaning was successful or no cleanup needed, False if errors occurred

##### get_timestamped_filename

```python
def get_timestamped_filename(
    self, 
    base_name: str, 
    extension: str = "json"
) -> str
```

Generate a timestamped filename.

**Parameters:**
- `base_name`: Base name for the file
- `extension`: File extension (without dot)

**Returns:**
- Timestamped filename

##### validate_directory_structure

```python
def validate_directory_structure(self) -> Dict[str, bool]
```

Validate the task directory structure.

**Returns:**
- Dictionary mapping directory types to validation results

##### list_artifacts

```python
def list_artifacts(
    self, 
    subdir: str = "output", 
    pattern: str = "*"
) -> List[Path]
```

List artifacts in a specific subdirectory.

**Parameters:**
- `subdir`: Subdirectory to search (e.g., "output", "visualizations")
- `pattern`: Glob pattern for filtering files

**Returns:**
- List of paths to matching artifacts

**Raises:**
- `DirectoryManagerError`: If subdirectory does not exist

##### import_external_file

```python
def import_external_file(
    self,
    source_path: Union[str, Path],
    subdir: str = "input",
    new_name: Optional[str] = None
) -> Path
```

Import an external file into the task directory structure.

**Parameters:**
- `source_path`: Path to the source file
- `subdir`: Target subdirectory (e.g., "input", "dictionaries")
- `new_name`: New name for the file (optional)

**Returns:**
- Path to the imported file

**Raises:**
- `PathValidationError`: If source path fails security validation
- `DirectoryManagerError`: If import fails

##### normalize_and_validate_path

```python
def normalize_and_validate_path(
    self, 
    path: Union[str, Path]
) -> Path
```

Normalize a path and validate its security.

**Parameters:**
- `path`: Path to normalize and validate

**Returns:**
- Normalized Path object

**Raises:**
- `PathValidationError`: If the path fails security validation

##### cleanup

```python
def cleanup(self) -> bool
```

Explicitly clean up resources.

**Returns:**
- True if cleanup was successful, False otherwise

#### Context Manager Support

The class implements the context manager protocol (`__enter__` and `__exit__`), allowing it to be used with the `with` statement:

```python
with TaskDirectoryManager(task_config) as dir_manager:
    # Use dir_manager here
    # temp directory will be cleaned on exit if configured
```

#### Internal Methods

##### _resolve_task_dir

```python
def _resolve_task_dir(self) -> Path
```

Resolve the task directory path from configuration.

**Returns:**
- Path to the task directory

**Raises:**
- `DirectoryManagerError`: If task directory cannot be resolved

##### _ensure_directories_with_progress

```python
def _ensure_directories_with_progress(self) -> Dict[str, Path]
```

Create and validate directories with progress tracking.

##### _ensure_directories_standard

```python
def _ensure_directories_standard(self) -> Dict[str, Path]
```

Create and validate directories without progress tracking.

##### _clean_temp_directory_with_progress

```python
def _clean_temp_directory_with_progress(self, temp_dir: Path) -> bool
```

Clean temporary directory with progress tracking.

##### _clean_temp_directory_standard

```python
def _clean_temp_directory_standard(self, temp_dir: Path) -> bool
```

Clean temporary directory without progress tracking.

## Helper Function

### create_directory_manager

```python
def create_directory_manager(
    task_config: Any,
    logger: Optional[logging.Logger] = None,
    progress_manager: Optional[TaskProgressManager] = None,
    initialize: bool = True
) -> TaskDirectoryManager
```

Create a directory manager for a task.

**Parameters:**
- `task_config`: Task configuration object
- `logger`: Logger for directory operations (optional)
- `progress_manager`: Progress manager for tracking directory operations (optional)
- `initialize`: Whether to initialize directories immediately

**Returns:**
- TaskDirectoryManager instance

**Raises:**
- `DirectoryManagerError`: If directory manager creation fails

## Directory Structure

The standard directory structure created by the directory manager is:

```
{task_dir}/
├── input/         # For input files
├── output/        # For output files
├── temp/          # For temporary files
├── logs/          # For log files
├── dictionaries/  # For data dictionaries
├── visualizations/ # For charts and visual artifacts
└── metrics/       # For metric outputs
```

Additional directories can be specified in the task configuration through the `task_dir_suffixes` attribute.

## Usage Examples

### Basic Directory Management

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Mock configuration
class MockConfig:
    task_id = "t_1A_profiling"
    project_root = Path("/path/to/project")
    task_dir_suffixes = ["input", "output", "temp", "logs", "custom_dir"]
    
    def get_task_dir(self):
        return self.project_root / "DATA" / "processed" / self.task_id
    
    def get_reports_dir(self):
        return self.project_root / "DATA" / "reports"

# Create directory manager
dir_manager = create_directory_manager(
    task_config=MockConfig(),
    logger=logger
)

# Get directory paths
input_dir = dir_manager.get_directory("input")
output_dir = dir_manager.get_directory("output")
custom_dir = dir_manager.get_directory("custom_dir")

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Custom directory: {custom_dir}")

# Clean temporary directory
dir_manager.clean_temp_directory()
```

### Creating Artifact Paths

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create directory manager
dir_manager = create_directory_manager(task_config, logger)

# Generate paths for different artifact types
profile_path = dir_manager.get_artifact_path(
    artifact_name="field_profile",
    artifact_type="json",
    subdir="output",
    include_timestamp=True
)

chart_path = dir_manager.get_artifact_path(
    artifact_name="distribution_chart",
    artifact_type="png",
    subdir="visualizations"
)

metrics_path = dir_manager.get_artifact_path(
    artifact_name="performance_metrics",
    artifact_type="csv",
    subdir="metrics"
)

print(f"Profile path: {profile_path}")
print(f"Chart path: {chart_path}")
print(f"Metrics path: {metrics_path}")
```

### Importing External Files

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager, PathValidationError
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create directory manager
dir_manager = create_directory_manager(task_config, logger)

try:
    # Import an external file to the input directory
    imported_path = dir_manager.import_external_file(
        source_path="/data/external/reference_data.csv",
        subdir="input"
    )
    print(f"Imported file to: {imported_path}")
    
    # Import and rename a file
    imported_dict = dir_manager.import_external_file(
        source_path="/data/external/dict.json",
        subdir="dictionaries",
        new_name="field_dictionary.json"
    )
    print(f"Imported and renamed file to: {imported_dict}")
    
except PathValidationError as e:
    print(f"Path validation error: {e}")
    # Handle security validation error
    
except DirectoryManagerError as e:
    print(f"Directory manager error: {e}")
    # Handle other directory-related errors
```

### Listing and Managing Artifacts

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create directory manager
dir_manager = create_directory_manager(task_config, logger)

# List all JSON artifacts in the output directory
json_files = dir_manager.list_artifacts(subdir="output", pattern="*.json")
print(f"Found {len(json_files)} JSON files:")
for file_path in json_files:
    print(f"- {file_path.name}")

# List all visualization artifacts
visualizations = dir_manager.list_artifacts(subdir="visualizations")
print(f"Found {len(visualizations)} visualization files")

# Validate the directory structure
validation_results = dir_manager.validate_directory_structure()
for dir_type, exists in validation_results.items():
    status = "exists" if exists else "missing"
    print(f"Directory '{dir_type}': {status}")
```

### Using with Progress Manager

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager
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

# Create directory manager with progress tracking
dir_manager = create_directory_manager(
    task_config=task_config,
    logger=logger,
    progress_manager=progress_manager,
    initialize=True  # This will show progress during initialization
)

# Directory operations will show progress
with progress_manager.create_operation_context(
    name="manage_directories",
    total=3,
    description="Managing directories"
) as progress:
    # Get artifact path
    path = dir_manager.get_artifact_path("result", "json")
    progress.update(1, {"path": str(path)})
    
    # Import external file with progress tracking
    imported = dir_manager.import_external_file("/data/external/data.csv", "input")
    progress.update(1, {"imported": str(imported)})
    
    # Clean temp directory with progress tracking
    cleaned = dir_manager.clean_temp_directory()  # This will use progress tracking internally
    progress.update(1, {"cleaned": cleaned})
```

### Normalizing and Validating Paths

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager, PathValidationError
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Create directory manager
dir_manager = create_directory_manager(task_config, logger)

try:
    # Normalize and validate a relative path
    relative_path = "input/data.csv"
    normalized_path = dir_manager.normalize_and_validate_path(relative_path)
    print(f"Normalized path: {normalized_path}")
    
    # Normalize and validate an absolute path
    absolute_path = "/data/processed/t_1A_profiling/output/result.json"
    normalized_absolute = dir_manager.normalize_and_validate_path(absolute_path)
    print(f"Normalized absolute path: {normalized_absolute}")
    
    # This should fail validation (path traversal attempt)
    unsafe_path = "../../../etc/passwd"
    try:
        normalized_unsafe = dir_manager.normalize_and_validate_path(unsafe_path)
    except PathValidationError as e:
        print(f"Unsafe path rejected: {e}")
    
except PathValidationError as e:
    print(f"Path validation error: {e}")
```

### Using as a Context Manager

```python
from pamola_core.utils.tasks.directory_manager import create_directory_manager
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("task.example")

# Configuration with clean_temp_on_exit enabled
class CleanupConfig:
    task_id = "t_1A_profiling"
    project_root = Path("/path/to/project")
    clean_temp_on_exit = True  # This will trigger temp cleanup on exit
    
    def get_task_dir(self):
        return self.project_root / "DATA" / "processed" / self.task_id
    
    def get_reports_dir(self):
        return self.project_root / "DATA" / "reports"

# Use directory manager as a context manager
with create_directory_manager(
    task_config=CleanupConfig(),
    logger=logger
) as dir_manager:
    
    # Create some temp files
    temp_dir = dir_manager.get_directory("temp")
    for i in range(5):
        temp_file = temp_dir / f"temp_file_{i}.tmp"
        with open(temp_file, "w") as f:
            f.write(f"Temporary content {i}")
    
    # Do work with temporary files
    # ...
    
    print("Work completed with temporary files")
    
# When exiting the with block, temporary files will be cleaned up automatically
```

## Integration with BaseTask

The `directory_manager.py` module is designed to integrate with the `BaseTask` class through the facade pattern. Here's how it's typically used within a task:

```python
# In BaseTask.initialize()
self.directory_manager = create_directory_manager(
    task_config=self.config,
    logger=self.logger,
    progress_manager=self.progress_manager
)

# Ensure directories are created
self.directories = self.directory_manager.ensure_directories()
self.task_dir = self.directory_manager.get_directory("task")

# In BaseTask methods
input_dir = self.directory_manager.get_directory("input")
output_dir = self.directory_manager.get_directory("output")

# Generate artifact paths
report_path = self.directory_manager.get_artifact_path(
    artifact_name="profile_report",
    artifact_type="html"
)

# Clean up in finalize()
self.directory_manager.clean_temp_directory()
```

## Path Resolution Process

The directory manager uses the following process to resolve the task directory:

1. Try to get task directory from configuration using `get_task_dir()` method
2. If that fails, try accessing the `task_dir` attribute directly
3. If that fails, try constructing from `processed_data_path` attribute
4. If all else fails, construct a default path from project root: `project_root/DATA/processed/{task_id}`

Once the task directory is resolved, subdirectories are created based on `task_dir_suffixes` or the default suffixes.

## Directory Structure Validation

The `validate_directory_structure()` method checks that all required directories exist and are accessible. This is useful for verifying that the directory structure is intact before proceeding with operations.

```python
validation_results = dir_manager.validate_directory_structure()
for dir_type, exists in validation_results.items():
    if not exists:
        logger.warning(f"Directory '{dir_type}' is missing")
```

## Artifact Path Generation

The `get_artifact_path()` method generates standardized paths for artifacts with optional timestamping. The generated path follows this pattern:

```
{task_dir}/{subdir}/{artifact_name}_{timestamp}.{artifact_type}
```

If `include_timestamp` is False:

```
{task_dir}/{subdir}/{artifact_name}.{artifact_type}
```

## Security Considerations

All paths are validated through the `validate_path_security()` function to prevent path traversal attacks. This includes:

1. Task directory path validation
2. Subdirectory path validation
3. Artifact path validation
4. External file path validation

Attempts to use paths that fail validation will raise a `PathValidationError`.

## Best Practices

1. **Use Standard Directory Types**: Stick to the standard directory types (input, output, temp, etc.) when possible for consistency.

2. **Clean Temporary Files**: Always clean the temporary directory when the task is complete to avoid accumulating unused files.

3. **Use Timestamped Artifacts**: Include timestamps in artifact filenames to avoid overwriting files and provide chronological context.

4. **Validate External Paths**: Always validate paths from external sources before using them.

5. **Use the Context Manager**: When possible, use the directory manager as a context manager to ensure proper cleanup.

6. **Include Progress Tracking**: When performing operations on many files, use progress tracking for better user experience.

7. **Use `get_artifact_path()`**: Always use this method instead of constructing paths manually to ensure consistency and security.

8. **Pattern Match Carefully**: When using `list_artifacts()` with patterns, be specific to avoid processing unintended files.

9. **Handle Validation Errors**: Always catch and handle `PathValidationError` exceptions for better error reporting.

10. **Import External Files**: Use `import_external_file()` instead of direct file operations to ensure proper path validation and tracking.