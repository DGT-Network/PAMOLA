# LLM Configuration Helpers Documentation

**Module:** `pamola_core.utils.nlp.llm.config_helpers`
**Version:** 1.2.0
**Last Updated:** 2026-03-23

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Dependencies](#dependencies)
4. [Core Classes](#core-classes)
5. [Utility Functions](#utility-functions)
6. [Usage Examples](#usage-examples)
7. [Best Practices](#best-practices)
8. [Related Components](#related-components)

## Overview

The `config_helpers` module provides high-level task configuration management for LLM operations. It builds upon the base configuration classes in `config.py` to deliver task-specific configuration handling, CLI argument merging, and model change detection with automatic cache management.

This module is designed to handle the complete lifecycle of task configuration from creation through updates, with support for configuration persistence, versioning, and change tracking. It is thread-safe for concurrent operations.

## Key Features

- **TaskConfig Dataclass**: Comprehensive configuration combining all LLM task aspects
- **ConfigManager**: Lifecycle management for configuration (load, save, track changes)
- **CLI Argument Merging**: Seamless integration of command-line arguments into configuration
- **Model Change Detection**: Automatic detection of model changes with optional cache clearing
- **Configuration Persistence**: Save and load configurations as JSON with history tracking
- **Thread-Safe Operations**: Lock-based thread safety for cache and history operations
- **Path Resolution**: Intelligent handling of absolute and relative paths
- **Configuration Diffing**: Track configuration changes between versions

## Dependencies

| Module | Purpose |
|--------|---------|
| `pamola_core.utils.nlp.llm.config` | Base configuration classes (LLMConfig, ProcessingConfig, etc.) |
| `pamola_core.utils.nlp.cache` | Cache management utilities |
| `pamola_core.errors.codes` | Error code definitions |
| `pamola_core.errors.exceptions` | Custom exception classes |
| `argparse` | Command-line argument parsing |
| `json` | Configuration serialization |
| `pathlib` | File path handling |
| `threading` | Thread synchronization primitives |
| `dataclasses` | Configuration data structures |
| `datetime` | Timestamp handling |

## Core Classes

### TaskPaths

Container for all file system paths used in task execution.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `data_repository` | `Path` | Root directory for all data |
| `dataset_path` | `Path` | Path to input dataset file |
| `task_dir` | `Path` | Task working directory |
| `output_dir` | `Path` | Directory for output files |
| `checkpoint_dir` | `Path` | Directory for model/processing checkpoints |
| `reports_dir` | `Path` | Directory for generated reports |

**Methods:**

- `__post_init__()`: Converts string paths to Path objects automatically
- `ensure_directories()`: Creates all necessary directories with parents

**Example:**

```python
from pathlib import Path
from pamola_core.utils.nlp.llm.config_helpers import TaskPaths

paths = TaskPaths(
    data_repository=Path("/data"),
    dataset_path=Path("/data/input.csv"),
    task_dir=Path("/data/processed/task1"),
    output_dir=Path("/data/processed/task1/output"),
    checkpoint_dir=Path("/data/processed/task1/checkpoints"),
    reports_dir=Path("/data/reports")
)

paths.ensure_directories()  # Creates all directories
```

### ColumnConfig

Configuration for dataframe column mapping during processing.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | — | Source column name to read from |
| `target` | `str` | — | Target column name to write to |
| `id_column` | `str` | `None` | Optional ID column for record tracking |
| `error_column` | `str` | `None` | Optional column for error logging |
| `backup_suffix` | `str` | `"_original"` | Suffix for backup columns |

**Properties:**

- `is_in_place` (bool): Returns True if source equals target (in-place mode)

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import ColumnConfig

# Out-of-place mode (safe)
columns = ColumnConfig(
    source="text",
    target="text_processed",
    id_column="record_id"
)

# In-place mode (destructive)
columns_inplace = ColumnConfig(source="text", target="text")
assert columns_inplace.is_in_place  # True
```

### DataConfig

Configuration for data processing behavior and file handling.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `encoding` | `str` | `"UTF-16"` | File encoding (e.g., UTF-8, UTF-16) |
| `separator` | `str` | `","` | CSV column separator |
| `text_qualifier` | `str` | `'"'` | Character for quoted fields |
| `start_id` | `int` | `None` | Starting record ID for partial processing |
| `end_id` | `int` | `None` | Ending record ID for partial processing |
| `max_records` | `int` | `None` | Maximum records to process |
| `create_backup` | `bool` | `True` | Create backup before processing |
| `warn_on_in_place` | `bool` | `True` | Warn when processing in-place |
| `require_existing_dataset` | `bool` | `True` | Require dataset to exist at validation |

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import DataConfig

config = DataConfig(
    encoding="UTF-8",
    separator="\t",
    max_records=1000,
    create_backup=True
)
```

### RuntimeConfig

Configuration for runtime behavior and error handling.

**Attributes:**

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `dry_run` | `bool` | `False` | Run without saving results |
| `test_connection_critical` | `bool` | `False` | Make connection test failures critical |
| `clear_cache_on_model_change` | `bool` | `True` | Clear cache when model changes |
| `max_errors` | `int` | `5` | Maximum allowed errors before stopping |
| `error_threshold` | `float` | `0.2` | Error rate threshold (0.0-1.0) |
| `force_reprocess` | `bool` | `False` | Force reprocessing of all records |
| `clear_target` | `bool` | `False` | Clear target column before processing |

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import RuntimeConfig

config = RuntimeConfig(
    dry_run=False,
    max_errors=10,
    error_threshold=0.15,
    clear_cache_on_model_change=True
)
```

### TaskConfig

Complete task configuration combining all configuration aspects. This is the main configuration class used throughout the module.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | Unique task identifier |
| `project_root` | `Path` | Project root directory |
| `llm` | `LLMConfig` | LLM connection configuration |
| `processing` | `ProcessingConfig` | Processing behavior settings |
| `generation` | `GenerationConfig` | Text generation parameters |
| `cache` | `CacheConfig` | Cache configuration |
| `monitoring` | `MonitoringConfig` | Monitoring and logging settings |
| `columns` | `ColumnConfig` | Column mapping configuration |
| `paths` | `TaskPaths` | File system paths |
| `data` | `DataConfig` | Data processing settings |
| `runtime` | `RuntimeConfig` | Runtime behavior settings |
| `prompt` | `dict` | Prompt template configuration |
| `metadata` | `dict` | Task metadata (created_at, version, etc.) |

**Key Methods:**

#### from_dict(config_dict)

Create TaskConfig from dictionary with validation.

```python
TaskConfig.from_dict(config_dict: Dict[str, Any]) -> TaskConfig
```

**Parameters:**
- `config_dict`: Configuration dictionary (may include legacy field formats)

**Returns:** Validated TaskConfig instance

**Raises:** `ConfigurationError` if required fields missing

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import TaskConfig

config_dict = {
    "task_id": "task_001",
    "project_root": "/workspace",
    "llm": {"provider": "lmstudio", "model_name": "LLM1"},
    "processing": {"batch_size": 1},
    "generation": {"temperature": 0.3},
    "cache": {"enabled": True},
    "monitoring": {},
    "columns": {"source": "text", "target": "text_processed"},
    "paths": {
        "data_repository": "/data",
        "dataset_path": "input.csv"
    }
}

config = TaskConfig.from_dict(config_dict)
```

#### merge_cli_args(args)

Merge command-line arguments into configuration.

```python
merge_cli_args(args: argparse.Namespace) -> TaskConfig
```

**Parameters:**
- `args`: Parsed command-line arguments from argparse

**Returns:** New TaskConfig with merged arguments

**Supported CLI Arguments:**
- `--model`: LLM model name
- `--start_id`, `--end_id`, `--max_records`: Data selection
- `--skip_processed`, `--no_skip_processed`: Processing behavior
- `--dry_run`, `--force_reprocess`, `--clear_target`: Runtime options
- `--no_cache`: Disable caching
- `--in_place`: Enable in-place mode
- `--no_backup`, `--create_backup`: Backup behavior
- `--debug_llm`, `--debug_log_file`: Debug options

**Example:**

```python
import argparse
from pamola_core.utils.nlp.llm.config_helpers import TaskConfig

config = TaskConfig.from_dict({...})

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="LLM1")
parser.add_argument("--max_records", type=int)
parser.add_argument("--dry_run", action="store_true")

args = parser.parse_args(["--model", "LLM2", "--max_records", "100"])
updated_config = config.merge_cli_args(args)
```

#### to_dict()

Convert to dictionary for serialization.

```python
to_dict() -> Dict[str, Any]
```

Returns dictionary with relative paths (when possible) for portability.

#### to_resolved_dict()

Convert to dictionary with fully resolved absolute paths.

```python
to_resolved_dict() -> Dict[str, Any]
```

Returns dictionary with all paths converted to absolute paths.

#### validate()

Validate configuration completeness and consistency.

```python
validate() -> None
```

**Raises:** `ConfigurationError` or `ValidationError` if invalid

**Checks:**
- Data repository exists
- Dataset exists (if required)
- Source and target columns specified
- Prompt template provided

### ConfigManager

Manages the complete lifecycle of task configuration with thread-safe operations.

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `task_id` | `str` | Task identifier |
| `project_root` | `Path` | Project root directory |
| `config_dir` | `Path` | Configuration directory |
| `config_path` | `Path` | Main configuration file path |

**Key Methods:**

#### get_default_config()

Get default configuration template for the task.

```python
get_default_config() -> Dict[str, Any]
```

Returns default configuration with sensible defaults. Override in subclasses for task-specific defaults.

#### load_config(reset, ignore, default_config)

Load configuration with fallback to defaults.

```python
load_config(
    reset: bool = False,
    ignore: bool = False,
    default_config: Optional[Dict[str, Any]] = None
) -> TaskConfig
```

**Parameters:**
- `reset`: Reset to default configuration and backup existing
- `ignore`: Use default configuration, ignoring saved config
- `default_config`: Custom default configuration dictionary

**Returns:** Loaded TaskConfig instance

**Behavior:**
- If `ignore=True`: Uses default configuration
- If file doesn't exist or `reset=True`: Creates new from defaults
- Otherwise: Loads from existing configuration file

**Example:**

```python
from pathlib import Path
from pamola_core.utils.nlp.llm.config_helpers import ConfigManager

manager = ConfigManager(task_id="task_001", project_root=Path("/workspace"))

# Load existing or use default
config = manager.load_config()

# Reset to default (backs up existing)
config = manager.load_config(reset=True)

# Use custom defaults
custom_defaults = {"task_id": "task_001", ...}
config = manager.load_config(default_config=custom_defaults)
```

#### save_config(config, suffix)

Save configuration to file.

```python
save_config(config: TaskConfig, suffix: Optional[str] = None)
```

**Parameters:**
- `config`: TaskConfig instance to save
- `suffix`: Optional suffix (e.g., "_backup") to create alternate filename

Automatically detects changes and records in history if configuration changed.

**Example:**

```python
config = manager.load_config()
config.runtime.dry_run = True

manager.save_config(config)  # Saves main config
manager.save_config(config, suffix="_backup")  # Saves as config_backup.json
```

#### check_model_change(current_model, task_dir, clear_on_change)

Check if model changed and optionally clear cache.

```python
check_model_change(
    current_model: str,
    task_dir: Path,
    clear_on_change: bool = True
) -> bool
```

**Parameters:**
- `current_model`: Current model name or alias
- `task_dir`: Task directory for model tracking
- `clear_on_change`: Clear cache if model changed

**Returns:** True if model changed, False otherwise

**Example:**

```python
from pathlib import Path

manager = ConfigManager("task_001", Path("/workspace"))
task_dir = Path("/workspace/processed/task_001")

if manager.check_model_change("LLM2", task_dir, clear_on_change=True):
    print("Model changed! Cache cleared.")
```

#### get_config_history()

Get configuration change history (thread-safe).

```python
get_config_history() -> list
```

**Returns:** List of configuration change records

#### save_config_change(change_type, details)

Record configuration change in history (thread-safe).

```python
save_config_change(change_type: str, details: Dict[str, Any])
```

**Parameters:**
- `change_type`: Type of change (e.g., "model_change", "reset", "cli_override")
- `details`: Change details dictionary

Automatically timestamped. Keeps last 100 changes.

**Example:**

```python
manager.save_config_change(
    "custom_change",
    {"reason": "user_request", "details": {...}}
)
```

## Utility Functions

### merge_nested_dicts(base, update)

Recursively merge nested dictionaries.

```python
merge_nested_dicts(
    base: Dict[str, Any],
    update: Dict[str, Any]
) -> Dict[str, Any]
```

**Returns:** Merged dictionary with update values overriding base values

### validate_task_config(config, required_fields)

Validate that configuration has required fields.

```python
validate_task_config(
    config: Dict[str, Any],
    required_fields: list
) -> None
```

**Parameters:**
- `config`: Configuration dictionary
- `required_fields`: List of required field paths (e.g., ["llm.model_name"])

**Raises:** `ConfigurationError` if required fields missing

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import validate_task_config

config = {...}
validate_task_config(config, [
    "task_id",
    "llm.model_name",
    "columns.source",
    "columns.target"
])
```

### get_field_from_path(config, field_path, default)

Get nested field value using dot notation.

```python
get_field_from_path(
    config: Dict[str, Any],
    field_path: str,
    default: Any = None
) -> Any
```

**Parameters:**
- `config`: Configuration dictionary
- `field_path`: Dot-separated path (e.g., "llm.model_name")
- `default`: Default value if field not found

**Returns:** Field value or default

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import get_field_from_path

config = {
    "llm": {"model_name": "LLM1"},
    "columns": {"source": "text"}
}

model = get_field_from_path(config, "llm.model_name")  # "LLM1"
missing = get_field_from_path(config, "unknown.field", "default")  # "default"
```

### set_field_from_path(config, field_path, value)

Set nested field value using dot notation.

```python
set_field_from_path(
    config: Dict[str, Any],
    field_path: str,
    value: Any
) -> Dict[str, Any]
```

**Parameters:**
- `config`: Configuration dictionary
- `field_path`: Dot-separated path (e.g., "llm.model_name")
- `value`: Value to set

**Returns:** Updated configuration dictionary

**Example:**

```python
from pamola_core.utils.nlp.llm.config_helpers import set_field_from_path

config = {"llm": {"model_name": "LLM1"}}
updated = set_field_from_path(config, "llm.temperature", 0.7)
# Result: {"llm": {"model_name": "LLM1", "temperature": 0.7}}
```

## Usage Examples

### Complete Configuration Workflow

```python
from pathlib import Path
from pamola_core.utils.nlp.llm.config_helpers import (
    TaskConfig,
    ConfigManager,
    TaskPaths,
    ColumnConfig,
    DataConfig,
    RuntimeConfig
)

# Initialize manager
manager = ConfigManager("nlp_task_001", Path("/workspace"))

# Load configuration
config = manager.load_config()

# Modify configuration
config.runtime.dry_run = True
config.data.max_records = 100

# Save configuration
manager.save_config(config)

# Check model changes
if manager.check_model_change("LLM2", config.paths.task_dir):
    print("Model changed. Cache cleared.")

# Get change history
history = manager.get_config_history()
for change in history:
    print(f"{change['timestamp']}: {change['type']}")
```

### CLI Integration

```python
import argparse
from pamola_core.utils.nlp.llm.config_helpers import ConfigManager
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--model")
    parser.add_argument("--max_records", type=int)
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    manager = ConfigManager(args.task_id, Path("."))
    config = manager.load_config()

    # Merge CLI arguments
    config = config.merge_cli_args(args)

    # Validate
    config.validate()

    # Save merged configuration
    manager.save_config(config)

    return config

if __name__ == "__main__":
    config = main()
```

## Best Practices

1. **Always validate configuration** after loading or modifying
2. **Use relative paths** in configuration files for portability
3. **Enable backups** when processing critical data
4. **Check for model changes** at task startup to detect issues
5. **Use ConfigManager** for all configuration operations to maintain consistency
6. **Merge CLI arguments early** in task initialization pipeline
7. **Track configuration history** for debugging and auditing
8. **Use absolute paths** in resolved dictionaries for deployment

## Related Components

- **`pamola_core.utils.nlp.llm.config`**: Base configuration classes (LLMConfig, ProcessingConfig, etc.)
- **`pamola_core.utils.nlp.cache`**: Cache management and retrieval
- **`pamola_core.errors.exceptions`**: Custom exception handling
- **`pamola_core.utils.nlp.llm.client`**: LLM client using this configuration
