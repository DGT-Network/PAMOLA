# Configuration Settings Module Documentation

**Module:** `pamola_core.configs.settings`
**Version:** 1.0
**Last Updated:** 2026-03-23

## Table of Contents
1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Configuration Structure](#configuration-structure)
4. [Loading Mechanism](#loading-mechanism)
5. [Data Repository Management](#data-repository-management)
6. [Singleton Pattern](#singleton-pattern)
7. [Usage Examples](#usage-examples)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Technical Summary](#technical-summary)

## Overview

The `settings.py` module provides core configuration management for PAMOLA.CORE applications. It handles loading configuration from multiple sources, managing a singleton configuration instance, and providing programmatic access to settings.

### Purpose

This module:
- Loads configuration from JSON files with intelligent file discovery
- Auto-detects data repository location in project structure
- Manages configuration with singleton pattern (one instance per process)
- Supports hierarchical configuration (nested dictionaries)
- Validates configuration parameters before use
- Integrates with environment variables for deployment flexibility
- Provides convenience functions for accessing specific settings

### Module Constants

```python
_CATALOG_DIR = Path(__file__).parent
_config = None  # Singleton instance
```

## Core Functions

### load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]

Loads configuration from file or uses defaults.

**Parameters:**
- `config_path` (str | Path, optional): Path to configuration JSON file

**Returns:**
- `Dict[str, Any]`: Configuration dictionary

**Behavior:**
1. Returns existing singleton if already loaded
2. Starts with DEFAULT_CONFIG
3. Loads from specified path (if provided)
4. Tries standard locations if no path specified
5. Auto-detects data repository if not in config
6. Applies logging configuration
7. Caches result in `_config` singleton

**Raises:**
- No exceptions; gracefully handles all errors

**Example:**
```python
from pamola_core.configs.settings import load_config

config = load_config()
config = load_config(config_path="custom_config.json")
```

### get_config() -> Dict[str, Any]

Retrieves the cached configuration (or loads if not yet loaded).

**Returns:**
- `Dict[str, Any]`: Current configuration

**Behavior:**
1. If `_config` is None, calls `load_config()`
2. Returns singleton instance
3. Guarantees non-None result (uses DEFAULT_CONFIG as fallback)

**Example:**
```python
from pamola_core.configs.settings import get_config

config = get_config()
print(f"Data repo: {config['data_repository']}")
```

### get_data_repository() -> Path

Gets the configured data repository path.

**Returns:**
- `Path`: Path to data repository

**Example:**
```python
from pamola_core.configs.settings import get_data_repository

repo = get_data_repository()
print(f"Data repository: {repo}")
```

### set_data_repository(path: Union[str, Path]) -> None

Explicitly sets the data repository path.

**Parameters:**
- `path` (str | Path): Path to data repository

**Behavior:**
1. Updates singleton configuration
2. Loads config if not yet loaded
3. Logs the change

**Example:**
```python
from pamola_core.configs.settings import set_data_repository

set_data_repository("/new/data/path")
```

### get_directory_structure() -> Dict[str, str]

Gets the directory structure configuration.

**Returns:**
- `Dict[str, str]`: Mapping of directory types to names

**Example:**
```python
from pamola_core.configs.settings import get_directory_structure

dirs = get_directory_structure()
print(f"Raw data: {dirs['raw']}")
print(f"Processed: {dirs['processed']}")
```

### get_performance_settings() -> Dict[str, Any]

Gets performance-related configuration.

**Returns:**
- `Dict[str, Any]`: Performance settings

**Example:**
```python
from pamola_core.configs.settings import get_performance_settings

perf = get_performance_settings()
print(f"Chunk size: {perf['chunk_size']}")
print(f"Memory limit: {perf['memory_limit_mb']}")
```

### get_logging_settings() -> Dict[str, Any]

Gets logging configuration.

**Returns:**
- `Dict[str, Any]`: Logging settings

**Example:**
```python
from pamola_core.configs.settings import get_logging_settings

logging_config = get_logging_settings()
print(f"Log level: {logging_config['level']}")
print(f"Log file: {logging_config['file']}")
```

### save_config(config_path: Optional[Union[str, Path]] = None) -> Path

Saves current configuration to JSON file.

**Parameters:**
- `config_path` (str | Path, optional): Path to save to

**Returns:**
- `Path`: Path to saved file

**Behavior:**
1. If path not specified, uses `{project_root}/configs/prj_config.json`
2. Creates parent directories if needed
3. Writes JSON with pretty-printing and UTF-8 encoding
4. Logs the operation

**Example:**
```python
from pamola_core.configs.settings import save_config

# Save with default path
path = save_config()

# Save with custom path
path = save_config(config_path="custom_location.json")
```

### update_nested_dict(d1: Dict[str, Any], d2: Dict[str, Any]) -> Dict[str, Any]

Deep merges two dictionaries.

**Parameters:**
- `d1` (Dict[str, Any]): Base dictionary
- `d2` (Dict[str, Any]): Update dictionary

**Returns:**
- `Dict[str, Any]`: Merged dictionary

**Behavior:**
1. Recursively merges nested dictionaries
2. Overwrites scalar values from d2
3. Preserves d1 structure where d2 has no override

**Example:**
```python
from pamola_core.configs.settings import update_nested_dict

defaults = {
    "performance": {
        "chunk_size": 100000,
        "memory_limit_mb": 1000
    }
}

overrides = {
    "performance": {
        "chunk_size": 50000
    }
}

result = update_nested_dict(defaults, overrides)
# {
#     "performance": {
#         "chunk_size": 50000,     # Overridden
#         "memory_limit_mb": 1000  # Preserved
#     }
# }
```

### get_config_file_paths() -> List[Path]

Gets candidate configuration file paths in search order.

**Returns:**
- `List[Path]`: List of paths to try

**Search Order:**
1. `PAMOLA_CONFIG_PATH` environment variable (if set)
2. `{project_root}/configs/prj_config.json`
3. `~/.pamola_core/config.json` (user home)
4. `{project_root}/prj_config.json` (fallback)

**Example:**
```python
from pamola_core.configs.settings import get_config_file_paths

paths = get_config_file_paths()
for path in paths:
    print(f"Will try: {path}")
```

## Configuration Structure

### DEFAULT_CONFIG

Built-in default configuration:

```python
DEFAULT_CONFIG = {
    "data_repository": None,  # Will be auto-detected
    "directory_structure": {
        "raw": "raw",
        "processed": "processed",
        "logs": "logs",
        "configs": "configs"
    },
    "logging": {
        "level": "INFO",
        "file": "pamola_processing.log",
        "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    },
    "performance": {
        "chunk_size": 100000,
        "default_encoding": "utf-8",
        "default_delimiter": ",",
        "memory_limit_mb": 1000
    }
}
```

### Top-Level Keys

| Key | Type | Purpose |
|-----|------|---------|
| `data_repository` | str | Path to data directory |
| `directory_structure` | dict | Directory organization |
| `logging` | dict | Logging settings |
| `performance` | dict | Performance tuning parameters |

### Directory Structure

```python
"directory_structure": {
    "raw": "raw",              # Input data directory name
    "processed": "processed",  # Output data directory name
    "logs": "logs",            # Log files directory name
    "configs": "configs"       # Configuration files directory name
}
```

### Logging Configuration

```python
"logging": {
    "level": "INFO",      # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "file": "pamola_processing.log",  # Log file name
    "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
}
```

Log levels (in order of severity):
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages
- `ERROR`: Error messages
- `CRITICAL`: Critical errors

### Performance Configuration

```python
"performance": {
    "chunk_size": 100000,      # Batch processing size
    "default_encoding": "utf-8",  # Text file encoding
    "default_delimiter": ",",   # CSV delimiter
    "memory_limit_mb": 1000     # Memory constraint (MB)
}
```

## Loading Mechanism

### File Search Strategy

Configuration files are searched in this order:

1. **Explicit Path** (if provided to `load_config()`)
   ```python
   load_config("custom_config.json")
   ```

2. **Environment Variable** (`PAMOLA_CONFIG_PATH`)
   ```bash
   export PAMOLA_CONFIG_PATH=/etc/pamola/config.json
   ```

3. **Project Directory** (`{project_root}/configs/prj_config.json`)
   - Auto-detected project root
   - Standard location for development

4. **User Home** (`~/.pamola_core/config.json`)
   - User-specific configuration
   - Overrides for individual developers

5. **Project Root Fallback** (`{project_root}/prj_config.json`)
   - Last resort location

### Data Repository Discovery

If `data_repository` is not configured, auto-detection attempts:

1. **Environment Variable** (`PAMOLA_DATA_REPOSITORY`)
   ```bash
   export PAMOLA_DATA_REPOSITORY=/data/production
   ```

2. **Project Directory Scan**
   - Looks for `{project_root}/data/` directory
   - Looks for `{project_root}/DATA/` directory (case variation)

3. **Project Root Fallback**
   - Uses project root as last resort

### Logging Configuration Application

After loading, logging level is applied:

```python
if "logging" in config and "level" in config["logging"]:
    level_name = config["logging"]["level"].upper()
    try:
        level = getattr(logging, level_name)
        logger.setLevel(level)
    except AttributeError:
        logger.warning(f"Invalid log level: {level_name}. Using INFO.")
```

## Singleton Pattern

The module uses a singleton pattern to ensure one configuration instance:

```python
_config = None  # Global singleton

def load_config(...) -> Dict[str, Any]:
    global _config

    if _config is not None:
        return _config  # Return existing instance

    # Load and cache
    _config = ...
    return _config

def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config
```

### Benefits

- **Memory Efficient**: Single configuration object in memory
- **Consistency**: All code sees same configuration
- **Thread-Safe**: Python GIL protects assignment
- **Lazy Loading**: Configuration loaded on-demand

### Modification Safety

The singleton pattern means modifications affect all code:

```python
from pamola_core.configs.settings import get_config, set_data_repository

# Modify singleton
set_data_repository("/new/path")

# All subsequent calls see the change
config = get_config()
assert config['data_repository'] == "/new/path"
```

## Usage Examples

### Example 1: Basic Configuration Load

```python
from pamola_core.configs.settings import get_config

# Load configuration (tries standard locations)
config = get_config()

# Access settings
print(f"Data repository: {config['data_repository']}")
print(f"Log level: {config['logging']['level']}")
print(f"Chunk size: {config['performance']['chunk_size']}")
```

### Example 2: Load from Custom Path

```python
from pamola_core.configs.settings import load_config
from pathlib import Path

# Load from specific file
config_path = Path("/etc/pamola/production_config.json")
config = load_config(config_path=config_path)

print(f"Configuration loaded from: {config_path}")
```

### Example 3: Extract Specific Settings

```python
from pamola_core.configs.settings import (
    get_data_repository,
    get_logging_settings,
    get_performance_settings
)

# Get specific setting groups
repo = get_data_repository()
logging_cfg = get_logging_settings()
perf_cfg = get_performance_settings()

print(f"Repo: {repo}")
print(f"Log file: {logging_cfg['file']}")
print(f"Chunk size: {perf_cfg['chunk_size']}")
```

### Example 4: Deep Dictionary Merge

```python
from pamola_core.configs.settings import update_nested_dict

defaults = {
    "nested": {
        "a": 1,
        "b": 2
    },
    "top": 10
}

overrides = {
    "nested": {
        "a": 100
    }
}

merged = update_nested_dict(defaults, overrides)

print(merged)
# {
#     "nested": {
#         "a": 100,   # Overridden
#         "b": 2      # Preserved
#     },
#     "top": 10       # Preserved
# }
```

### Example 5: Modify and Save Configuration

```python
from pamola_core.configs.settings import (
    get_config,
    set_data_repository,
    save_config
)

# Load configuration
config = get_config()

# Modify settings
set_data_repository("/production/data")
config['performance']['chunk_size'] = 50000

# Save changes
config_path = save_config(config_path="updated_config.json")

print(f"Configuration saved to: {config_path}")
```

### Example 6: Directory Structure Configuration

```python
from pamola_core.configs.settings import get_directory_structure
from pathlib import Path

# Get directory structure
dirs = get_directory_structure()

# Build data paths
base_path = Path("/data")
raw_path = base_path / dirs['raw']
processed_path = base_path / dirs['processed']
logs_path = base_path / dirs['logs']

print(f"Raw data: {raw_path}")
print(f"Processed: {processed_path}")
print(f"Logs: {logs_path}")
```

## Best Practices

### 1. **Load Configuration Once at Startup**

```python
# Good: Load once at application start
from pamola_core.configs.settings import get_config

config = get_config()

# Pass configuration to components
def process_data(config):
    chunk_size = config['performance']['chunk_size']
    # Use configuration
```

### 2. **Use Convenience Functions**

```python
from pamola_core.configs.settings import (
    get_data_repository,
    get_logging_settings,
    get_performance_settings
)

# Good: Use domain-specific getters
repo = get_data_repository()
logging_cfg = get_logging_settings()

# Less clean: Direct dictionary access
config = get_config()
repo = config['data_repository']
```

### 3. **Handle Missing Data Repository**

```python
from pamola_core.configs.settings import get_config

config = get_config()

if config.get('data_repository') is None:
    raise ValueError(
        "Data repository not configured. "
        "Set PAMOLA_DATA_REPOSITORY environment variable or "
        "configure in prj_config.json"
    )
```

### 4. **Validate Configuration on Load**

```python
from pamola_core.configs.settings import get_config
from pathlib import Path

config = get_config()

# Validate data repository exists
repo = Path(config['data_repository'])
if not repo.exists():
    raise ValueError(f"Data repository does not exist: {repo}")

# Validate chunk size is reasonable
chunk_size = config['performance']['chunk_size']
if chunk_size < 1000:
    raise ValueError(f"Chunk size too small: {chunk_size}")
```

### 5. **Use Environment Variables for Deployment**

```bash
# Docker/Kubernetes environment
export PAMOLA_CONFIG_PATH=/etc/pamola/config.json
export PAMOLA_DATA_REPOSITORY=/mnt/data

# Python application
from pamola_core.configs.settings import get_config
config = get_config()  # Respects environment variables
```

## Troubleshooting

### Issue: Configuration Singleton Not Respecting Changes

**Problem:**
```python
from pamola_core.configs.settings import get_config, set_data_repository

set_data_repository("/path1")
config1 = get_config()

set_data_repository("/path2")
config2 = get_config()

# config1 and config2 are the same object
assert config1 is config2
assert config1['data_repository'] == "/path2"  # Both point to /path2
```

**Solution:**
Singleton pattern is intentional. All code shares same configuration. To use different configs:
- Restart application with different environment variables
- Load specific config before other code: `load_config(config_path)`

### Issue: Project Root Not Auto-Detected

**Problem:**
```
Error detecting data repository: [error message]. Using project root.
```

**Cause:**
- `get_project_root()` utility failed
- Project structure non-standard

**Solution:**
1. Explicitly set: `export PAMOLA_DATA_REPOSITORY=/path/to/data`
2. Create config file: `configs/prj_config.json`

### Issue: Configuration File Not Found

**Problem:**
```python
config = load_config("nonexistent.json")
# No error, but defaults used instead
```

**Cause:**
- Module doesn't raise error if file not found
- Falls back to defaults silently

**Solution:**
1. Verify file exists: `ls configs/prj_config.json`
2. Check file is readable: `cat configs/prj_config.json`
3. Validate JSON: `python -m json.tool configs/prj_config.json`

### Issue: Invalid JSON in Configuration File

**Problem:**
```
Warning: Failed to load config from [path]: [JSON error]
```

**Cause:**
- JSON syntax error in config file
- Unexpected characters or unclosed quotes

**Solution:**
Validate JSON:
```bash
python -m json.tool configs/prj_config.json
```

Fix syntax errors and retry.

## Technical Summary

The `settings.py` module provides robust configuration management:

- **Multi-Source Loading**: Files, environment variables, defaults
- **Smart Discovery**: Auto-detects data repository and project structure
- **Singleton Pattern**: One configuration instance per process
- **Hierarchical Support**: Nested dictionaries for organized settings
- **Deep Merge**: Proper override handling for complex structures
- **Environment Integration**: Full support for containerized deployments
- **Lazy Loading**: Configuration loaded on first access
- **Convenience Functions**: Domain-specific getters for common settings

The module abstracts configuration complexity while maintaining flexibility for diverse deployment scenarios.
