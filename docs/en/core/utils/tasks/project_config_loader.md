# ProjectConfigLoader Module Documentation

## Overview

The `project_config_loader.py` module provides functionality for loading and managing project-level configurations in the PAMOLA Core framework. It serves as the central mechanism for identifying project roots, loading configuration files, and standardizing access to project structure. This module supports YAML configuration with enhanced readability and comments, while maintaining backward compatibility with JSON configurations.

## Key Features

- **Project Root Discovery**: Automatically locates project root through multiple strategies
- **YAML Configuration**: Primary support for YAML with enhanced readability and comments
- **JSON Backward Compatibility**: Fallback support for legacy JSON configurations
- **Variable Substitution**: Dynamic variable replacement using Jinja2 templates
- **Configuration Caching**: Performance optimization through configuration caching
- **Default Values**: Automatic application of sensible defaults for missing configuration
- **Structured Validation**: Organized configuration structure with sectional defaults
- **Path Resolution**: Standardized access to project directory structure
- **Project Initialization**: Support for creating new project structures

## Dependencies

- `yaml`: For YAML parsing (PyYAML library)
- `json`: For JSON parsing (Standard library)
- `logging`: For logging support
- `pathlib.Path`: For path manipulation
- `jinja2.Template`: For variable substitution (optional)

## Constants and Global Variables

- `PROJECT_CONFIG_FILENAME`: Default YAML configuration filename (`"prj_config.yaml"`)
- `PROJECT_CONFIG_LEGACY_FILENAME`: Legacy JSON configuration filename (`"prj_config.json"`)
- `_config_cache`: Dictionary cache for parsed configurations
- `JINJA2_AVAILABLE`: Flag indicating whether Jinja2 template engine is available

## Key Functions

### find_project_root

```python
def find_project_root() -> Path
```

Locate the project root directory.

**Priority:**
1) PAMOLA_PROJECT_ROOT environment variable
2) .pamolaProject marker file or presence of configs/prj_config.{yaml,json}
3) Git repository root (if GitPython is available; otherwise search for .git directory)
4) Current directory (with warning)

**Returns:**
- Absolute path to the project root.

### substitute_variables

```python
def substitute_variables(config_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]
```

Perform variable substitution in configuration values using Jinja2.

**Parameters:**
- `config_data`: Configuration dictionary to process
- `context`: Dictionary of variables for substitution

**Returns:**
- Configuration with variables substituted

### load_project_config

```python
def load_project_config(
    project_root: Optional[Path] = None,
    config_filename: Optional[str] = None,
    use_cache: bool = True
) -> Dict[str, Any]
```

Load the project configuration from a YAML file with JSON fallback.

**Parameters:**
- `project_root`: Path to the project root directory. If None, it will be auto-detected.
- `config_filename`: Name of the configuration file. If None, uses default names.
- `use_cache`: Whether to use cached configuration if available.

**Returns:**
- Parsed configuration dictionary with defaults applied.

**Raises:**
- `FileNotFoundError`: If the configuration file doesn't exist.
- `ValueError`: If the configuration file is invalid.

### apply_default_values

```python
def apply_default_values(config_data: Dict[str, Any]) -> Dict[str, Any]
```

Apply default values to the configuration where values are missing.

**Parameters:**
- `config_data`: Original configuration dictionary

**Returns:**
- Configuration with defaults applied

### clear_config_cache

```python
def clear_config_cache() -> None
```

Clear the configuration cache.

### get_project_paths

```python
def get_project_paths(config: Dict[str, Any], project_root: Optional[Path] = None) -> Dict[str, Path]
```

Get standard project paths from configuration.

**Parameters:**
- `config`: Project configuration dictionary
- `project_root`: Project root path (auto-detected if None)

**Returns:**
- Dictionary of standard project paths

### save_project_config

```python
def save_project_config(
    config_data: Dict[str, Any],
    project_root: Optional[Path] = None,
    format: str = "yaml"
) -> Path
```

Save the project configuration to a file.

**Parameters:**
- `config_data`: Configuration dictionary to save
- `project_root`: Project root path (auto-detected if None)
- `format`: Format to save in - "yaml" or "json"

**Returns:**
- Path to saved configuration file

### is_valid_project_root

```python
def is_valid_project_root(path: Path) -> bool
```

Check if a path is a valid project root.

**Parameters:**
- `path`: Path to check

**Returns:**
- True if the path is a valid project root, False otherwise

### create_default_project_structure

```python
def create_default_project_structure(root_path: Path, data_path: Optional[Path] = None) -> Dict[str, Path]
```

Create a default project structure at the specified location.

**Parameters:**
- `root_path`: Root path for the new project
- `data_path`: Path for data repository (defaults to DATA under root_path)

**Returns:**
- Dictionary of created paths

### get_recursive_variables

```python
def get_recursive_variables(config_data: Dict[str, Any]) -> Dict[str, Any]
```

Extract all variables from configuration data that can be used in substitution.

**Parameters:**
- `config_data`: Configuration dictionary

**Returns:**
- Dictionary of flattened variables for substitution

## Default Configuration Structure

The module applies these default values to configurations:

```python
defaults = {
    "directory_structure": {
        "raw": "raw",
        "processed": "processed",
        "reports": "reports",
        "logs": "logs",
        "configs": "configs"
    },
    "task_dir_suffixes": [
        "input",
        "output",
        "temp",
        "logs",
        "dictionaries"
    ],
    "logging": {
        "level": "INFO",
    },
    "performance": {
        "chunk_size": 100000,
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "default_quotechar": "\"",
        "memory_limit_mb": 1000,
        "use_dask": False
    },
    "encryption": {
        "use_encryption": False,
        "encryption_mode": "none",
        "key_path": None
    },
    "task_defaults": {
        "continue_on_error": True,
        "parallel_processes": 4
    }
}
```

## Variable Substitution

The module supports variable substitution using Jinja2 templates. Variables available for substitution include:

1. `${project_root}`: Absolute path to project root
2. Any top-level configuration keys (e.g., `${data_repository}`)
3. Nested keys using dot notation (e.g., `${directory_structure.raw}`)

## Usage Examples

### Basic Configuration Loading

```python
from pamola_core.utils.tasks.project_config_loader import load_project_config
from pathlib import Path
import logging

# Set up logger
logger = logging.getLogger("project.config")

# Load project configuration
try:
    config = load_project_config()
    
    # Access configuration values
    project_root = config.get("project_root")
    data_repository = config.get("data_repository")
    logging_level = config.get("logging", {}).get("level", "INFO")
    
    print(f"Project root: {project_root}")
    print(f"Data repository: {data_repository}")
    print(f"Logging level: {logging_level}")
    
except FileNotFoundError:
    logger.error("Project configuration file not found")
except ValueError as e:
    logger.error(f"Invalid project configuration: {e}")
```

### Working with Project Paths

```python
from pamola_core.utils.tasks.project_config_loader import load_project_config, get_project_paths
from pathlib import Path

# Load project configuration
config = load_project_config()

# Get standard project paths
paths = get_project_paths(config)

# Access paths
project_root = paths["project_root"]
data_repo = paths["data_repository"]
raw_dir = paths["raw_dir"]
processed_dir = paths["processed_dir"]
reports_dir = paths["reports_dir"]
logs_dir = paths["logs_dir"]
configs_dir = paths["configs_dir"]

print(f"Project root: {project_root}")
print(f"Data repository: {data_repo}")
print(f"Raw data directory: {raw_dir}")
print(f"Processed data directory: {processed_dir}")
print(f"Reports directory: {reports_dir}")
print(f"Logs directory: {logs_dir}")
print(f"Configs directory: {configs_dir}")
```

### Creating a New Project Structure

```python
from pamola_core.utils.tasks.project_config_loader import create_default_project_structure, save_project_config
from pathlib import Path

# Create new project structure
project_root = Path("/path/to/new_project")
data_path = Path("/data/external_repository")

# Create directories
dirs = create_default_project_structure(
    root_path=project_root,
    data_path=data_path
)

print("Created project structure:")
for name, path in dirs.items():
    print(f"  {name}: {path}")

# Customize configuration
config = {
    "project_root": str(project_root),
    "data_repository": str(data_path),
    "project_name": "My PAMOLA Project",
    "logging": {
        "level": "DEBUG",
        "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    },
    "encryption": {
        "use_encryption": True,
        "encryption_mode": "simple"
    }
}

# Save custom configuration
config_path = save_project_config(
    config_data=config,
    project_root=project_root,
    format="yaml"
)

print(f"Saved custom configuration to {config_path}")
```

### Using Variable Substitution

```python
from pamola_core.utils.tasks.project_config_loader import load_project_config
from pathlib import Path

# Create a custom configuration with variables
custom_config = {
    "project_root": "/path/to/project",
    "data_repository": "${project_root}/data",
    "output_dir": "${data_repository}/processed",
    "task_dir": "${output_dir}/${task_id}",
    "task_id": "t_1P1"
}

# Example of loading config with variable substitution
# (This is done automatically by load_project_config)
from pamola_core.utils.tasks.project_config_loader import substitute_variables

# Create context for substitution
context = {
    "project_root": "/path/to/project",
    "task_id": "t_1P1"
}

# Apply variable substitution
result = substitute_variables(custom_config, context)

# Print resolved values
print(f"Project root: {result['project_root']}")
print(f"Data repository: {result['data_repository']}")
print(f"Output directory: {result['output_dir']}")
print(f"Task directory: {result['task_dir']}")
```

### Handling Multiple Configuration Files

```python
from pamola_core.utils.tasks.project_config_loader import load_project_config, find_project_root, clear_config_cache
from pathlib import Path

# Find project root
project_root = find_project_root()

# Load main project configuration
main_config = load_project_config(
    project_root=project_root,
    config_filename="prj_config.yaml"
)

# Load environment-specific configuration (e.g., development, production)
env = "development"
env_config = load_project_config(
    project_root=project_root,
    config_filename=f"prj_config.{env}.yaml",
    use_cache=False  # Don't use cache for environment config
)

# Merge configurations (environment overrides main)
merged_config = {**main_config, **env_config}

# Clear cache if needed
clear_config_cache()
```

### Checking Valid Project Root

```python
from pamola_core.utils.tasks.project_config_loader import is_valid_project_root, find_project_root
from pathlib import Path

# Check if a specific path is a valid project root
path = Path("/path/to/check")
if is_valid_project_root(path):
    print(f"{path} is a valid PAMOLA project root")
else:
    print(f"{path} is not a valid PAMOLA project root")

# Find the nearest valid project root
nearest_root = find_project_root()
print(f"Nearest project root: {nearest_root}")
```

## Integration with Other Modules

The `project_config_loader.py` module is designed to integrate with other PAMOLA Core modules:

### Integration with BaseTask

```python
# In BaseTask.initialize()
from pamola_core.utils.tasks.project_config_loader import load_project_config, get_project_paths

def initialize(self, args: Optional[Dict[str, Any]] = None) -> bool:
    try:
        # Load configuration
        self.config = load_task_config(self.task_id, self.task_type, args)
        
        # Initialize project paths
        self.project_paths = get_project_paths(self.config)
        
        # Set up logging
        self._setup_logging()
        
        # ...continue with initialization
        return True
    except Exception as e:
        # Handle initialization error
        return False
```

### Integration with TaskDirectoryManager

```python
# In TaskDirectoryManager
from pamola_core.utils.tasks.project_config_loader import get_project_paths

def __init__(self, task_config: Any, logger: Optional[logging.Logger] = None):
    self.config = task_config
    self.logger = logger
    
    # Get project paths
    self.paths = get_project_paths(self.config)
    
    # Access standard paths
    self.project_root = self.paths["project_root"]
    self.data_repository = self.paths["data_repository"]
    # ...continue with initialization
```

### Integration with TaskConfigLoader

```python
# In task_config.py
from pamola_core.utils.tasks.project_config_loader import load_project_config, find_project_root

def load_task_config(task_id: str, task_type: str, args: Optional[Dict[str, Any]] = None) -> Any:
    # Find project root
    project_root = find_project_root()
    
    # Load project-level config
    project_config = load_project_config(project_root)
    
    # Load task-specific config
    # ...
    
    # Merge configurations with correct precedence
    # ...
    
    return merged_config
```

## Project Structure

The default project structure created by the module is:

```
{project_root}/
├── .pamolaProject          # Marker file
├── configs/                # Configuration files
│   └── prj_config.yaml     # Project configuration
├── logs/                   # Log files
├── pamola_core/                   # PAMOLA Core framework
├── scripts/                # Task scripts
└── data/                   # Data repository (can be external)
    ├── raw/                # Raw input data
    ├── processed/          # Processed data by tasks
    └── reports/            # Task execution reports
```

## Project Root Discovery Process

The module uses the following algorithm to discover the project root:

1. **Environment Variable**: Check `PAMOLA_PROJECT_ROOT` environment variable
   - If set and points to a valid directory, use it
   - Valid if it contains `.pamolaProject` marker file or `configs` directory

2. **Marker File or Configs Directory**: Search up from current directory
   - Look for `.pamolaProject` marker file
   - Or look for `configs/prj_config.yaml` or `configs/prj_config.json`
   - Search up to a maximum depth (defined by `PAMOLA_MAX_SEARCH_DEPTH` env var or default 20)

3. **Git Repository Root**: Try to find Git repository root
   - Use GitPython if available
   - Otherwise, search for `.git` directory manually
   - Search up to maximum depth

4. **Fallback**: Use current working directory
   - Log a warning
   - Suggest setting `PAMOLA_PROJECT_ROOT` or creating `.pamolaProject` marker

## Configuration Formats

### YAML Configuration (Preferred)

```yaml
# Project configuration
project_root: "D:/VK/_DEVEL/PAMOLA.CORE"
data_repository: "D:/VK/_DEVEL/PAMOLA.CORE/data"

# Directory structure
directory_structure:
  raw: "raw"
  processed: "processed"
  reports: "reports"
  logs: "logs"
  configs: "configs"

# Standard task directory suffixes
task_dir_suffixes:
  - "input"
  - "output"
  - "temp"
  - "dictionaries"

# Logging configuration
logging:
  level: "INFO"
  format: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

# Default performance settings
performance:
  chunk_size: 100000
  default_encoding: "utf-8"
  default_delimiter: ","
  memory_limit_mb: 1000
  use_dask: false

# Default encryption settings
encryption:
  use_encryption: false
  encryption_mode: "none"  # Accepted values: none, simple, fernet, age
  key_path: null

# Task defaults
task_defaults:
  continue_on_error: true
  parallel_processes: 4

# Task-specific configuration overrides
tasks:
  t_1I:
    description: "Initial data ingestion"
    dependencies: []
  t_1P1:
    description: "Group profiling"
    dependencies: ["t_1I"]
```

### JSON Configuration (Legacy Support)

```json
{
  "project_root": "D:/VK/_DEVEL/PAMOLA.CORE",
  "data_repository": "D:/VK/_DEVEL/PAMOLA.CORE/data",
  "directory_structure": {
    "raw": "raw",
    "processed": "processed",
    "reports": "reports",
    "logs": "logs",
    "configs": "configs"
  },
  "task_dir_suffixes": [
    "input",
    "output",
    "temp",
    "dictionaries"
  ],
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
  },
  "performance": {
    "chunk_size": 100000,
    "default_encoding": "utf-8",
    "default_delimiter": ",",
    "memory_limit_mb": 1000,
    "use_dask": false
  },
  "encryption": {
    "use_encryption": false,
    "encryption_mode": "none",
    "key_path": null
  },
  "task_defaults": {
    "continue_on_error": true,
    "parallel_processes": 4
  },
  "tasks": {
    "t_1I": {
      "description": "Initial data ingestion",
      "dependencies": []
    },
    "t_1P1": {
      "description": "Group profiling",
      "dependencies": ["t_1I"]
    }
  }
}
```

## Best Practices

1. **Use YAML Format**: YAML is more readable and supports comments

2. **Use Variables**: Use variable substitution to keep configurations DRY

3. **Project Root Marker**: Always create a `.pamolaProject` marker file at project root

4. **Environment Variables**: Consider using environment variables for sensitive information

5. **Default Values**: Let the framework provide default values for common parameters

6. **Separate Environments**: Use environment-specific configuration files for different environments

7. **Version Control**: Include project configuration in version control

8. **Clear Cache**: Clear configuration cache when configuration changes

9. **Validate Paths**: Validate paths from configuration before use

10. **Standardize Directories**: Follow the standard directory structure convention

## Limitations and Considerations

1. **Jinja2 Dependency**: Variable substitution requires Jinja2 library (optional dependency)

2. **GitPython Dependency**: Git repository detection works best with GitPython (optional dependency)

3. **Configuration Size**: Very large configurations may impact performance

4. **Circular References**: Variable substitution doesn't handle circular references

5. **Environment Variables**: Sensitive information in configurations may be exposed in logs

6. **Case Sensitivity**: Path handling may differ between operating systems

7. **Relative Paths**: Relative paths are resolved against project root, which may cause confusion

8. **Configuration Updates**: Updates to configuration files require cache clearing

## Tips for Troubleshooting

If you're having trouble with configuration loading:

1. **Check Environment Variables**: Verify `PAMOLA_PROJECT_ROOT` is set correctly if used

2. **Validate YAML Syntax**: Ensure your YAML configuration is properly formatted

3. **Clear Cache**: Call `clear_config_cache()` to force config reload

4. **Check Project Structure**: Verify your project has the expected structure

5. **Debug Logging**: Enable debug logging to see configuration loading details

```python
import logging
logging.getLogger("pamola_core.utils.tasks.project_config_loader").setLevel(logging.DEBUG)
```

6. **Inspect Loaded Config**: Print the loaded configuration to verify all values

```python
import pprint
pprint.pprint(load_project_config())
```

7. **Create Marker File**: Add a `.pamolaProject` file at your project root

8. **Verify File Permissions**: Ensure configuration files are readable