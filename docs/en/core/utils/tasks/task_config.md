# TaskConfig Module Documentation

## Overview

The `task_config.py` module provides comprehensive functionality for loading, managing, and validating task configurations in the PAMOLA Core framework. It implements a cascading configuration system that merges settings from multiple sources with clear precedence rules. The module handles project navigation, path resolution, and ensures secure and consistent configuration across all tasks in the PAMOLA ecosystem.

## Key Features

- **Project Navigation**: Reliably locates project root directory and data repositories
- **Configuration Cascade**: Implements priority chain (CLI args → task JSON → default config + project config overrides)
- **Bootstrap Mechanism**: Automatically creates task-specific configuration files when they don't exist
- **Hierarchical Configuration**: Supports project-level settings with task-specific overrides
- **Path Resolution**: Resolves both absolute and relative paths with security validation
- **Directory Structure Management**: Defines and maintains standardized task directory hierarchies
- **Environment Variable Support**: Integrates with environment variables for configuration overrides
- **Path API**: Provides a clean API for accessing standard task directories
- **Configuration Persistence**: Saves and updates configuration files in JSON or YAML formats
- **Caching**: Caches resolved paths for performance optimization
- **Security Validation**: Prevents path traversal and other security issues 
- **Dependency Management**: Handles task dependencies and validation
- **Progress Integration**: Optional integration with progress tracking

## Dependencies

- `logging`: Standard Python logging functionality
- `os`: Operating system interface for environment variables
- `pathlib.Path`: Object-oriented filesystem path handling
- `enum.Enum`: Enumeration support for encryption modes
- `yaml`: YAML parsing and serialization (for project configuration)
- `json`: JSON parsing and serialization (for task configuration)
- `pamola_core.utils.io`: File I/O utilities for reading/writing configuration files
- `pamola_core.utils.tasks.path_security`: Path security validation
- `pamola_core.utils.tasks.project_config_loader`: Project configuration loading
- `pamola_core.utils.tasks.progress_manager`: Optional progress tracking

## Configuration Loading Sequence

The module implements a specific configuration loading sequence designed to balance standardization with flexibility:

1. **Project YAML Loading**
   - Load the global project configuration from `configs/prj_config.yaml`
   - Extract project-wide settings (directory structure, defaults, logging)
   - Extract task-specific overrides from `tasks.{task_id}` section

2. **Task JSON Check**
   - Check if task-specific JSON exists at `configs/{task_id}.json`
   
   - **If JSON exists**:
     - Load and use EXCLUSIVELY this JSON for task configuration
     - This is the normal path for established tasks
     - Changes to the project YAML's `tasks.{task_id}` section will NOT affect the task
   
   - **If JSON doesn't exist** (Bootstrap Mode):
     - Start with code-defined defaults from `task.get_default_config()`
     - Apply project YAML overrides from `tasks.{task_id}` section on top
     - Save the combined configuration as `configs/{task_id}.json`
     - This only happens on the first run of a task

3. **Subsequent Runs**
   - After first run, the task-specific JSON always exists
   - All subsequent runs will use ONLY the JSON file
   - To re-apply defaults, the task JSON file must be manually deleted

This sequence ensures that:
- Default configurations are applied only once when creating a new task
- Each task maintains its own isolated configuration after initial setup
- Project-level changes won't unexpectedly affect existing tasks

The final configuration priority is:
1. Command-line arguments (highest)
2. Task-specific JSON configuration (from `configs/{task_id}.json`)
3. For first run only (if JSON doesn't exist):
   - Default configuration from task code + Project YAML's `tasks.{task_id}` overrides

## Class: EncryptionMode

### Description

Enumeration of encryption modes supported by the task framework.

### Values

- `NONE`: No encryption
- `SIMPLE`: Simple symmetric encryption
- `AGE`: Age encryption (more secure, supports key rotation)

### Methods

#### from_string

```python
@classmethod
def from_string(cls, value: str) -> 'EncryptionMode'
```

Converts a string to an EncryptionMode enum value.

**Parameters:**
- `value`: String representation of the encryption mode

**Returns:**
- Corresponding EncryptionMode enum value

**Behavior:**
- Handles case-insensitive conversion
- Falls back to SIMPLE mode with warning if value is invalid

## Exceptions

### ConfigurationError

Exception raised for configuration-related errors such as invalid format, missing required parameters, or file access issues.

### DependencyMissingError

Exception raised when required task dependencies are not satisfied, such as when a dependency task has not been executed or its output is not available.

### PathSecurityError

Exception raised when a path fails security validation, such as attempting to access directories outside the project boundary.

## Class: TaskConfig

### Description

Task configuration container and manager that holds configuration parameters and provides methods for accessing them with the correct priority cascade.

### Constructor

```python
def __init__(self,
             config_dict: Dict[str, Any],
             task_id: str,
             task_type: str,
             env_override: bool = True,
             progress_manager: Optional[TaskProgressManager] = None)
```

**Parameters:**
- `config_dict`: Dictionary containing configuration values
- `task_id`: ID of the task this configuration is for
- `task_type`: Type of the task this configuration is for
- `env_override`: Whether to allow environment variables to override configuration
- `progress_manager`: Optional progress manager for tracking initialization

### Key Attributes

- `task_id`: ID of the task
- `task_type`: Type of task (profiling, anonymization, etc.)
- `project_root`: Absolute path to project root directory
- `data_repository`: Path to data repository (relative or absolute)
- `data_repository_path`: Absolute path to data repository
- `log_level`: Logging level for the task
- `task_dir`: Base directory for the task
- `task_dir_suffixes`: List of standard suffixes for task directories
- `dependencies`: List of task IDs that this task depends on
- `continue_on_error`: Whether to continue execution if an operation fails
- `use_encryption`: Whether to use encryption for sensitive data
- `encryption_mode`: Encryption mode to use (EncryptionMode enum)
- `encryption_key_path`: Path to the encryption key file
- `output_directory`: Path to the task output directory
- `input_dir`: Path to task input directory
- `output_dir`: Path to task output directory
- `temp_dir`: Path to task temporary directory
- `logs_dir`: Path to task logs directory
- `dictionaries_dir`: Path to task dictionaries directory
- `report_path`: Path to the task report
- `log_file`: Path to the log file
- `task_log_file`: Path to the task-specific log file
- `scope`: Configuration for operation scope (fields, datasets, field groups)

**Performance settings:**
- `use_vectorization`: Whether to use vectorized (parallel) processing
- `parallel_processes`: Number of parallel processes to use
- `chunk_size`: Size of data chunks for processing
- `memory_limit_mb`: Memory limit in megabytes
- `use_dask`: Whether to use Dask for large dataset processing
- `default_encoding`: Default character encoding for file operations
- `default_delimiter`: Default delimiter for CSV files
- `default_quotechar`: Default quote character for CSV files

**Security settings:**
- `allow_external`: Whether to allow access to external paths
- `allowed_external_paths`: List of external paths that are allowed to be accessed
- `legacy_path_support`: Whether to support legacy path resolution

### Methods

#### Path API Methods

```python
def get_project_root() -> Path
```
Get the project root directory.

```python
def get_data_repository() -> Path
```
Get the data repository path.

```python
def get_raw_dir() -> Path
```
Get the raw data directory.

```python
def get_processed_dir() -> Path
```
Get the processed data directory.

```python
def get_reports_dir() -> Path
```
Get the reports directory.

```python
def get_task_dir(task_id: Optional[str] = None) -> Path
```
Get the task directory for the specified task ID or the current task.

```python
def get_task_input_dir(task_id: Optional[str] = None) -> Path
```
Get the task input directory for the specified task ID or the current task.

```python
def get_task_output_dir(task_id: Optional[str] = None) -> Path
```
Get the task output directory for the specified task ID or the current task.

```python
def get_task_temp_dir(task_id: Optional[str] = None) -> Path
```
Get the task temporary directory for the specified task ID or the current task.

```python
def get_task_dict_dir(task_id: Optional[str] = None) -> Path
```
Get the task dictionaries directory for the specified task ID or the current task.

```python
def get_task_logs_dir(task_id: Optional[str] = None) -> Path
```
Get the task logs directory for the specified task ID or the current task.

```python
def processed_subdir(task_id: Optional[str] = None, *parts) -> Path
```
Get a subdirectory within the processed directory for a specific task.

#### Dependency Management Methods

```python
def get_dependency_output(dependency_id: str, file_pattern: Optional[str] = None) -> Union[Path, List[Path]]
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

```python
def get_dependency_report(dependency_id: str) -> Path
```
Get the report file for a dependency.

**Parameters:**
- `dependency_id`: Dependency ID (task ID)

**Returns:**
- Path to the dependency report file

**Raises:**
- `DependencyMissingError`: If the dependency report doesn't exist

```python
def assert_dependencies_completed() -> bool
```
Check if all dependencies have completed successfully.

**Returns:**
- True if all dependencies are complete, False otherwise

**Raises:**
- `DependencyMissingError`: If a dependency report is missing or indicates failure

#### Scope Methods

```python
def get_scope_fields() -> List[str]
```
Get fields defined in the scope.

**Returns:**
- List of field names in the scope

```python
def get_scope_datasets() -> List[str]
```
Get datasets defined in the scope.

**Returns:**
- List of dataset names in the scope

```python
def get_scope_field_groups() -> Dict[str, List[str]]
```
Get field groups defined in the scope.

**Returns:**
- Dictionary mapping group names to lists of field names

#### Configuration Management Methods

```python
def override_with_args(self, args: Dict[str, Any]) -> None
```
Override configuration with command-line arguments.

**Parameters:**
- `args`: Command-line arguments dictionary

```python
def validate(self) -> Tuple[bool, List[str]]
```
Validate the configuration.

**Returns:**
- Tuple containing:
  - Boolean indicating whether configuration is valid
  - List of validation error messages

```python
def to_dict(self) -> Dict[str, Any]
```
Convert configuration to dictionary.

**Returns:**
- Dictionary representation of configuration

```python
def save(self, path: Optional[Path] = None, format: str = "json") -> Path
```
Save configuration to file.

**Parameters:**
- `path`: Path to save configuration file, or None to use default
- `format`: Format to save in - "json" or "yaml"

**Returns:**
- Path to saved configuration file

```python
def resolve_legacy_path(self, path: Union[str, Path]) -> Path
```
Resolve a path using legacy format during transition period.

**Parameters:**
- `path`: Path in legacy format

**Returns:**
- Resolved absolute path

**Behavior:**
- Issues deprecation warning for legacy path usage
- Validates path security
- Returns absolute path

## Functions

### load_task_config

```python
def load_task_config(
    task_id: str,
    task_type: str,
    args: Optional[Dict[str, Any]] = None,
    default_config: Optional[Dict[str, Any]] = None,
    progress_manager: Optional[TaskProgressManager] = None
) -> TaskConfig
```

Loads task configuration following the configuration cascade logic.

**Parameters:**
- `task_id`: ID of the task
- `task_type`: Type of the task
- `args`: Command-line arguments to override configuration
- `default_config`: Default configuration from task class (used during bootstrap)
- `progress_manager`: Optional progress manager for tracking configuration loading

**Returns:**
- TaskConfig instance with loaded configuration

**Raises:**
- `ConfigurationError`: If configuration cannot be loaded or validated

**Behavior:**
1. Loads project configuration from YAML
2. Checks for task-specific JSON:
   - If exists: Uses exclusively this JSON for task configuration
   - If doesn't exist: 
     - Uses defaults from task class
     - Applies project YAML overrides
     - Saves to new JSON file
3. Creates TaskConfig instance
4. Applies command-line overrides
5. Validates configuration

## Directory Structure

The module manages the standard PAMOLA directory structure:

```
PROJECT_ROOT/
├── configs/                      # Configuration files
│   ├── prj_config.yaml           # Project configuration (YAML)
│   └── {task_id}.json            # Task configuration (JSON)
├── logs/                         # Log files
│   └── {task_id}.log             # Task-specific log
└── DATA/                         # Data repository
    ├── raw/                      # Raw input data
    ├── processed/                # Processed data
    │   └── {task_id}/            # Task-specific directory
    │       ├── input/            # Task inputs
    │       ├── output/           # Task outputs
    │       ├── temp/             # Temporary files
    │       ├── logs/             # Task-specific logs
    │       └── dictionaries/     # Extracted dictionaries
    └── reports/                  # Task reports
        └── {task_id}_report.json # Task report
```

## Project Root Discovery

The module provides several methods to discover the project root:
1. Environment variable: `PAMOLA_PROJECT_ROOT`
2. Marker files/directories: Searches upward for `.pamola`, `configs`, `DATA`
3. Git repository root: Uses Git repository root if available
4. Fallback: Uses current directory if no markers found

## Security Validation

The module includes security validation to prevent path traversal and other security issues:
- Validates paths using the `path_security` module
- Checks for parent directory traversal (`..`)
- Validates absolute paths against allowed external paths
- Provides separate exception type for security violations
- Controls external path access with `allow_external` and `allowed_external_paths`

## Environment Variables

The module supports environment variables with the prefix `PAMOLA_`:
- `PAMOLA_PROJECT_ROOT`: Path to project root
- `PAMOLA_DATA_REPOSITORY`: Path to data repository
- `PAMOLA_LOG_LEVEL`: Default log level
- `PAMOLA_TASK_{TASK_ID}_{KEY}`: Task-specific settings

Environment variables are converted to appropriate types:
- Boolean: 'true', 'yes', '1', 'on' → True; 'false', 'no', '0', 'off' → False
- None: 'none', 'null' → None
- Integer: Numeric strings without decimal points
- Float: Numeric strings with decimal points
- List: Comma-separated values
- String: Default for all other values

## Task Default Configuration

Task classes can provide default configuration through the `get_default_config()` method:

```python
def get_default_config(self) -> Dict[str, Any]:
    """Get default configuration values for this task."""
    return {
        "description": "Task description",
        "dependencies": ["dependency_task_id"],
        "use_vectorization": True,
        "parallel_processes": 4,
        "fields": ["field1", "field2", "field3"],
        "task_specific_setting": "value"
    }
```

These defaults are used only during the first run when no task-specific JSON file exists. After the first run, the task's JSON file becomes the exclusive source of task configuration.

## Progress Tracking

The module supports optional progress tracking during configuration loading:
1. Finding project root
2. Loading project configuration
3. Loading/creating task-specific configuration
4. Creating TaskConfig instance
5. Validating configuration

## Usage Example

```python
from pamola_core.utils.tasks.task_config import load_task_config

# Define task defaults
def get_default_config():
    return {
        "description": "Profile dataset for privacy risks",
        "dependencies": ["t_1I"],
        "continue_on_error": True,
        "fields": ["name", "age", "address", "ssn"],
        "use_vectorization": True
    }

# Load configuration for a task
config = load_task_config(
    task_id="t_1P1_profile",
    task_type="profiling",
    default_config=get_default_config(),
    args={"log_level": "DEBUG"}
)

# Access configuration using Path API
project_root = config.get_project_root()
task_dir = config.get_task_dir()
output_dir = config.get_task_output_dir()

print(f"Task ID: {config.task_id}")
print(f"Project root: {project_root}")
print(f"Task directory: {task_dir}")
print(f"Output directory: {output_dir}")

# Access dependency outputs
dependency_outputs = config.get_dependency_output("t_1I", "*.csv")
for output_file in dependency_outputs:
    print(f"Dependency output file: {output_file}")

# Get scope configuration
fields = config.get_scope_fields()
datasets = config.get_scope_datasets()
field_groups = config.get_scope_field_groups()

print(f"Fields in scope: {fields}")
print(f"Datasets in scope: {datasets}")
print(f"Field groups: {field_groups}")

# Save updated configuration
config.save()
```

## Best Practices

1. **Provide Task Defaults**: Always implement `get_default_config()` in task classes to ensure consistent initial configuration.

2. **Use Path API**: Always use the provided Path API methods (`get_task_dir()`, `get_task_output_dir()`, etc.) instead of constructing paths manually.

3. **Configuration Isolation**: Understand that each task maintains its own isolated JSON configuration after initial setup.

4. **Reset to Defaults**: To reset a task to defaults, delete its JSON file and run the task again.

5. **Task-Specific Overrides**: For project-wide changes that should affect existing tasks, edit each task's JSON file directly.

6. **Dependency Management**: Use the dependency management methods to access outputs from dependency tasks rather than hardcoding paths.

7. **Environment Variables**: Use environment variables for deployment-specific settings and CI/CD pipelines.

8. **Security Validation**: Always validate paths before using them, especially when dealing with user inputs.

9. **Configuration Persistence**: Use `save()` method to persist configuration changes rather than modifying files directly.

10. **Progress Tracking**: Pass a progress manager to `load_task_config()` for better user feedback during initialization.

11. **Validation**: Check validation results to ensure configuration is valid before proceeding with task execution.

12. **Legacy Path Support**: Avoid using `resolve_legacy_path()` in new code - use the Path API instead.